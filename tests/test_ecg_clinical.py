"""The structured clinical reading must never over-claim.

Three properties matter more than the formatting: an unmeasurable parameter
gets no verdict, a barely-readable strip is not called NORMAL, and a named
abnormality is only named when the measurements support it.
"""
import pytest

from ecg.clinical import (clinical_report, parameter_table, diagnostic_status,
                          precautions, DISCLAIMER)
from ecg.quality import Measurement, OK, UNAVAILABLE


def _report(**vals):
    """Build a report; values in native units (seconds for intervals)."""
    keys = ["heart_rate", "rhythm", "pr_interval", "qrs_duration",
            "qt_interval", "qtc", "st_segment", "axis"]
    out = {}
    for k in keys:
        if k in vals:
            v = vals[k]
            out[k] = (v if isinstance(v, Measurement) else Measurement(v, OK))
        else:
            out[k] = Measurement(None, UNAVAILABLE, "not detected")
    return out


NORMAL = dict(heart_rate=72.0, pr_interval=0.16, qrs_duration=0.09,
              qtc=0.41, st_segment=0.0,
              rhythm=Measurement(0.03, OK, "Regular"))


# ------------------------------------------------- unmeasurable parameters

def test_an_unmeasurable_parameter_gets_no_verdict():
    rows = {r["key"]: r for r in parameter_table(_report())}
    assert all(r["verdict"] is None for r in rows.values())
    assert all(r["value"] == "Not measurable" for r in rows.values())
    assert rows["heart_rate"]["reason"]


def test_a_low_quality_measurement_is_not_treated_as_measured():
    from ecg.quality import LOW
    rows = {r["key"]: r
            for r in parameter_table(_report(heart_rate=Measurement(72.0, LOW)))}
    assert rows["heart_rate"]["verdict"] is None


def test_every_row_states_a_formula_and_a_reference_range():
    for row in parameter_table(_report(**NORMAL)):
        assert row["formula"] and row["range"]


# ------------------------------------------------------ diagnostic status

def test_a_clean_tracing_is_normal():
    rep = _report(**NORMAL)
    status = diagnostic_status(rep, parameter_table(rep))
    assert status["classification"] == "NORMAL"
    assert status["abnormality"] == "Normal sinus rhythm"
    assert status["findings"] == []


def test_an_unreadable_tracing_is_indeterminate_not_normal():
    """The single most dangerous possible output is calling an unreadable
    strip normal."""
    rep = _report(heart_rate=72.0)
    status = diagnostic_status(rep, parameter_table(rep))
    assert status["classification"] == "INDETERMINATE"
    assert status["classification"] != "NORMAL"


@pytest.mark.parametrize("field,value,expected", [
    ("heart_rate", 45.0, "Bradycardia"),
    ("heart_rate", 130.0, "Tachycardia"),
    ("pr_interval", 0.26, "Prolonged PR interval"),
    ("qrs_duration", 0.16, "Wide QRS complex"),
    ("qtc", 0.50, "Prolonged QTc"),
    ("st_segment", 3.0, "ST elevation"),
])
def test_named_abnormalities_follow_the_measurement(field, value, expected):
    rep = _report(**{**NORMAL, field: value})
    status = diagnostic_status(rep, parameter_table(rep))
    assert status["classification"] == "ABNORMAL"
    assert expected in status["abnormality"]


def test_a_wide_qrs_is_described_not_diagnosed_as_a_bundle_branch_block():
    """Morphology is needed to classify; this pipeline measures width only."""
    rep = _report(**{**NORMAL, "qrs_duration": 0.16})
    name = diagnostic_status(rep, parameter_table(rep))["abnormality"]
    assert "Bundle Branch Block" not in name
    assert "morphology" in name.lower()


def test_an_irregular_rhythm_is_reported_without_diagnosing_af():
    rep = _report(**{**NORMAL, "rhythm": Measurement(0.30, OK, "Irregular")})
    status = diagnostic_status(rep, parameter_table(rep))
    assert status["classification"] == "ABNORMAL"
    assert "Irregular" in status["abnormality"]
    assert "Atrial Fibrillation" not in status["abnormality"]


def test_qtc_ceiling_follows_the_recorded_sex():
    """0.45 s is normal for a female patient and prolonged for a male one."""
    rep = _report(**{**NORMAL, "qtc": 0.45})
    male = {r["key"]: r for r in parameter_table(rep, sex="Male")}
    female = {r["key"]: r for r in parameter_table(rep, sex="Female")}
    assert male["qtc"]["verdict"] == "Above range"
    assert female["qtc"]["verdict"] == "Within range"


def test_the_beat_classifier_contributes_a_finding():
    rep = _report(**NORMAL)
    status = diagnostic_status(rep, parameter_table(rep),
                               {"prediction": "ABNORMAL",
                                "abnormal_type": "Ventricular ectopic beat"})
    assert status["classification"] == "ABNORMAL"
    assert any("Ventricular ectopic" in f for f in status["findings"])


# ------------------------------------------------------------ precautions

def test_precautions_are_specific_to_what_was_found():
    rep = _report(**{**NORMAL, "st_segment": 3.0})
    status = diagnostic_status(rep, parameter_table(rep))
    text = " ".join(precautions(status))
    assert "myocardial infarction" in text
    assert "emergency" in text.lower()


def test_normal_advice_does_not_claim_disease_is_excluded():
    rep = _report(**NORMAL)
    text = " ".join(precautions(diagnostic_status(rep, parameter_table(rep))))
    assert "does not exclude" in text


def test_indeterminate_advice_says_it_reflects_quality_not_health():
    rep = _report(heart_rate=72.0)
    text = " ".join(precautions(diagnostic_status(rep, parameter_table(rep))))
    assert "not the absence of disease" in text


def test_the_full_report_always_carries_the_disclaimer():
    out = clinical_report(_report(**NORMAL), sex="Male")
    assert out["disclaimer"] == DISCLAIMER
    assert "not a diagnosis" in out["disclaimer"]
    assert set(out) == {"parameters", "status", "precautions", "disclaimer"}
