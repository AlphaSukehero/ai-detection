"""A result page must not contradict itself or print its own placeholders.

Found on a real page: a normal 72 bpm strip was headed ABNORMAL because its
QRS was narrow, advised "routine monitoring" and "cardiology review" in the
same list, showed "None%", and refilled the form with "Not provided".
"""
import re
from datetime import date

import pytest

import app as flask_app
from ecg.clinical import clinical_report, doctor_notes
from tests.conftest_records import rclient  # noqa: F401
from tests.test_ecg_clinical import NORMAL, _report
from tests.test_ecg_route import _post
from tests.test_routes import PATIENT, requires_mri_model

BLANK = {"patient_name": "", "patient_id": "", "age": "", "gender": "",
         "contact": "", "referring_physician": "", "study_date": "",
         "clinical_history": ""}


def _quick_ecg(client, sample="normal", **fields):
    data = dict(BLANK, sample_type=sample)
    data.update(fields)
    return client.post("/analyze_ecg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)


def _form_values(html, form_id):
    """{name: value} of the text inputs and textareas of one form."""
    form = re.search(rf'<form[^>]*id="{form_id}".*?</form>', html, re.S).group(0)
    values = dict(re.findall(r'<input type="(?:text|tel|date|number)"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form, re.S))
    values.update(re.findall(r'<textarea[^>]*name="([^"]+)"[^>]*>(.*?)</textarea>', form, re.S))
    return values


# ------------------------------------------------------------ verdict

def test_a_narrow_qrs_is_not_an_abnormal_finding():
    """Only a wide QRS is pathological; a narrow one is a normal complex."""
    clinical = clinical_report(_report(**{**NORMAL, "qrs_duration": 0.053}))
    row = next(r for r in clinical["parameters"] if r["key"] == "qrs_duration")
    assert row["verdict"] == "Within range"
    assert clinical["status"]["classification"] == "NORMAL"


def test_a_wide_qrs_is_still_flagged():
    clinical = clinical_report(_report(**{**NORMAL, "qrs_duration": 0.16}))
    assert clinical["status"]["classification"] == "ABNORMAL"


def test_normal_sample_strip_is_headed_normal():
    with flask_app.app.test_client() as c:
        html = _quick_ecg(c)
    verdict = re.search(r'ECG Classification</div>\s*<div[^>]*>\s*([A-Z ]+?)\s*</div>', html)
    assert verdict and verdict.group(1) == "NORMAL", verdict and verdict.group(1)


def test_advice_does_not_reassure_and_refer_in_one_list():
    """The classifier's 'routine monitoring' line has no place in the advice
    of a study the structured reading calls abnormal."""
    clinical = clinical_report(_report(**{**NORMAL, "heart_rate": 130.0}))
    notes = doctor_notes(
        clinical, "NORMAL", "The CNN classified the ECG as NORMAL.",
        "Normal rhythm detected. Routine clinical monitoring recommended.")
    assert notes["verdict"]["label"] == "ABNORMAL"
    advice = " ".join(notes["recommendations"])
    assert "Cardiology review" in advice
    assert "Routine clinical monitoring" not in advice


def test_agreeing_classifier_advice_is_kept():
    clinical = clinical_report(_report(**NORMAL))
    notes = doctor_notes(clinical, "NORMAL", None,
                         "Normal rhythm detected. Routine clinical monitoring recommended.")
    assert "Routine clinical monitoring" in " ".join(notes["recommendations"])


# ------------------------------------------------------------ placeholders

def test_no_page_prints_a_python_none():
    with flask_app.app.test_client() as c:
        for sample in ("normal", "abnormal"):
            html = _quick_ecg(c, sample)
            text = re.sub(r"<script.*?</script>|<[^>]+>", " ", html, flags=re.S)
            assert not re.search(r"\bNone\b", text), re.search(r".{40}\bNone\b.{20}", text).group(0)
            assert "nan" not in text.lower().split()


def test_blank_details_stay_blank_in_the_form():
    """'Not provided' is what the report prints for a blank, not something
    the patient typed: sent back, it becomes the patient's name."""
    with flask_app.app.test_client() as c:
        html = _quick_ecg(c)
    values = _form_values(html, "ecg-form")
    for key in ("patient_name", "patient_id", "contact", "referring_physician",
                "clinical_history"):
        assert values[key].strip() == "", (key, values[key])
    assert "Not provided" in html          # still shown in the study details


def test_typed_details_are_kept_in_the_form():
    with flask_app.app.test_client() as c:
        html = _post(c, "/analyze_ecg", {"sample_type": "normal"}).get_data(as_text=True)
    values = _form_values(html, "ecg-form")
    assert values["patient_name"] == PATIENT["patient_name"]
    assert values["study_date"] == PATIENT["study_date"]


def test_resubmitting_a_result_page_does_not_rename_the_patient(rclient):
    first = _quick_ecg(rclient, patient_id="11005")
    again = dict(BLANK, sample_type="normal", **_form_values(first, "ecg-form"))
    second = rclient.post("/analyze_ecg", data=again,
                          content_type="multipart/form-data").get_data(as_text=True)
    assert "Saved to record" in second
    assert "already on record as" not in second
    assert "Not provided" not in rclient.get("/patients").get_data(as_text=True)


def test_a_blank_study_date_is_today_on_the_page_and_in_the_record(rclient):
    today = date.today().isoformat()
    html = _quick_ecg(rclient, patient_id="11005")
    assert f'name="study_date" value="{today}"' in html
    sid = re.search(r"ST-\d{8}-\d{4}", html).group(0)
    pid = re.search(r"/patients/(PT-\d+)/", html).group(1)
    assert today in rclient.get(f"/patients/{pid}/studies/{sid}").get_data(as_text=True)


def test_upload_help_no_longer_describes_a_280_sample_file():
    with flask_app.app.test_client() as c:
        assert "280-sample" not in c.get("/ecg").get_data(as_text=True)
