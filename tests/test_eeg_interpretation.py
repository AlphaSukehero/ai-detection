"""Doctor-style wording for the EEG report: impression, advice, precautions."""
import pytest

from eeg.interpretation import clinical_notes

R = {"episodes_n": 2, "burden_pct": 12.5, "duration_s": 60.0,
     "episodes": [{"onset_s": 4.0, "duration_s": 6.0}],
     "band_means": {"delta": 0.5, "theta": 0.3, "alpha": 0.1, "beta": 0.08,
                    "gamma": 0.02}}


@pytest.mark.parametrize("task", ["seizure", "alzheimer"])
@pytest.mark.parametrize("label", ["ABNORMAL", "NORMAL", "NOT ASSESSED"])
def test_every_case_has_an_impression_advice_and_precautions(task, label):
    n = clinical_notes(task, label, R)
    assert n["impression"] and n["recommendations"] and n["precautions"]
    assert all(isinstance(x, str) and x for x in
               n["recommendations"] + n["precautions"])


def test_abnormal_seizure_gives_seizure_safety_precautions():
    n = clinical_notes("seizure", "ABNORMAL", R)
    text = " ".join(n["precautions"]).lower()
    assert "driv" in text and "5 minutes" in text
    assert "4.0" in n["impression"]  # cites the earliest onset


def test_abnormal_slowing_advises_cognitive_workup():
    n = clinical_notes("alzheimer", "ABNORMAL", R)
    assert "cognitive" in " ".join(n["recommendations"]).lower()


def test_normal_does_not_exclude_disease():
    n = clinical_notes("seizure", "NORMAL", R)
    assert "does not exclude" in n["impression"].lower()


def test_not_assessed_never_reads_as_normal():
    n = clinical_notes("seizure", "NOT ASSESSED", R)
    assert "normal" not in n["impression"].lower().replace("abnormal", "")


def test_missing_numbers_do_not_raise():
    assert clinical_notes("seizure", "ABNORMAL", {})["impression"]
