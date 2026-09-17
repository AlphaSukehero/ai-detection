"""Doctor-style wording for the brain MRI report."""
import pytest

from vision.interpretation import clinical_notes, mri_verdict


@pytest.mark.parametrize("label,expected", [
    ("Glioma", "ABNORMAL"), ("Meningioma", "ABNORMAL"),
    ("Pituitary", "ABNORMAL"), ("No Tumor", "NORMAL"), ("", "NOT ASSESSED")])
def test_verdict(label, expected):
    assert mri_verdict(label)["label"] == expected


@pytest.mark.parametrize("label", ["Glioma", "Meningioma", "Pituitary",
                                   "No Tumor", ""])
def test_every_class_has_impression_advice_and_precautions(label):
    n = clinical_notes(label, {"location": "Left frontal", "severity": "High",
                               "area": "4.2", "confidence": "91.0"})
    assert n["impression"] and n["recommendations"] and n["precautions"]


def test_tumour_impression_cites_the_measurements():
    n = clinical_notes("Glioma", {"location": "Left frontal", "area": "4.2",
                                  "confidence": "91.0"})
    assert "Left frontal" in n["impression"] and "91.0%" in n["impression"]


def test_pituitary_precautions_mention_vision():
    text = " ".join(clinical_notes("Pituitary", {})["precautions"]).lower()
    assert "vision" in text


def test_no_tumour_does_not_exclude_disease():
    assert "does not exclude" in clinical_notes("No Tumor", {})["impression"].lower()


def test_missing_values_do_not_raise():
    assert clinical_notes("Meningioma", {})["impression"]


@pytest.mark.parametrize("location", ["Central Middle region", "Left frontal"])
def test_location_is_not_double_suffixed(location):
    text = clinical_notes("Glioma", {"location": location})["impression"]
    assert "region area" not in text and "region region" not in text
    assert location in text
