"""The app must never present an invented number as a model result.

Three fabrication paths existed and were removed:

  * predict_ecg() fell back to an 8-qubit QCNN whose preprocessing was an
    unpickled PCA + MinMaxScaler fitted under a different scikit-learn minor
    version, and below that to sigmoid(std(X)) presented as a confidence;
  * fallback_mri_analysis() returned hardcoded probability tables
    ("78.4% Glioma") from a brightness threshold, which the report then
    dressed as "Urgent neurosurgical consultation recommended".

These tests exist so none of them can return quietly.
"""
import inspect

import numpy as np
import pytest

import app as flask_app


def test_there_is_no_mri_heuristic_fallback():
    assert not hasattr(flask_app, "fallback_mri_analysis")


def test_predict_brain_tumor_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(flask_app, "get_brain_tumor_model", lambda: (None, None))
    with pytest.raises(flask_app.NoMRIModelError):
        flask_app.predict_brain_tumor("static/samples/mri_glioma.jpg")


def test_predict_ecg_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(flask_app, "predict_ecg_cnn", lambda X: None)
    with pytest.raises(flask_app.NoECGModelError):
        flask_app.predict_ecg((np.zeros((1, 280)), np.ones((1, 4))))


def test_no_quantum_circuit_remains():
    """pennylane is no longer a runtime dependency of the app."""
    source = inspect.getsource(flask_app)
    assert "pennylane" not in source
    assert "qcnn_circuit" not in source


def test_no_pickle_load_of_preprocessing():
    """Cross-version unpickled estimators transform silently wrong."""
    source = inspect.getsource(flask_app)
    assert "pickle.load" not in source


def test_ecg_report_table_states_absence_instead_of_dashes():
    """With no classification, the PDF must say so, not print em-dashes that
    read like a missing measurement."""
    rows = flask_app._ecg_classification_rows({"classifier_error": "no model"})
    flat = dict(rows[1:])
    assert flat["Overall Rhythm Classification"] == "Not performed"
    assert flat["Reason"] == "no model"
    assert not any("%" in str(v) for v in flat.values())


def test_ecg_report_table_reports_a_real_classification():
    rows = flask_app._ecg_classification_rows({
        "prediction": "NORMAL", "abnormal_type": "No ectopic beats detected",
        "confidence": "91.20", "atrial_probability": "1.10",
        "classifier": "AAMI 5-class 1D CNN"})
    flat = dict(rows[1:])
    assert flat["Overall Rhythm Classification"] == "NORMAL"
    assert flat["Classifier Confidence"] == "91.20%"


# ------------------------- per-class reliability gating (item 1)

REAL_CARD = {
    "classes": ["N", "S", "V", "F", "Q"],
    "metrics": {"per_class_f1": {"N": 0.762, "S": 0.097, "V": 0.556,
                                 "F": 0.008, "Q": 0.003}},
}


def test_only_classes_the_model_can_discriminate_are_nameable():
    """S/F/Q sit at F1 0.097/0.008/0.003 -- naming them is invented detail."""
    assert flask_app._reliable_ecg_classes(REAL_CARD) == {"N", "V"}


def test_an_unmeasured_card_is_not_silently_trusted_or_silently_dropped(capsys):
    card = {"classes": ["N", "S"], "metrics": {}}
    assert flask_app._reliable_ecg_classes(card) == {"N", "S"}
    assert "no per-class F1" in capsys.readouterr().out


def test_abnormal_strip_is_not_given_a_name_the_model_cannot_support(monkeypatch):
    """An S-dominant strip must be called abnormal without claiming "S"."""
    import numpy as np

    class FakeModel:
        def predict(self, x, verbose=0):
            n = len(x)
            # every beat confidently S -- a class with F1 0.097
            p = np.tile(np.array([0.1, 0.8, 0.05, 0.03, 0.02]), (n, 1))
            return p

    monkeypatch.setattr(flask_app, "get_ecg_model",
                        lambda: (FakeModel(), REAL_CARD))
    res = flask_app.predict_ecg_cnn((np.zeros((4, 280)), np.ones((4, 4))))
    assert res["prediction"] == "ABNORMAL"
    assert "cannot" in res["abnormal_type"].lower()
    assert "Supraventricular" not in res["abnormal_type"]
    assert res["atrial_probability"] is None


def test_reliable_classes_are_still_named(monkeypatch):
    """V is at F1 0.556 -- suppression must not swallow what the model can do."""
    import numpy as np

    class FakeModel:
        def predict(self, x, verbose=0):
            return np.tile(np.array([0.1, 0.05, 0.8, 0.03, 0.02]), (len(x), 1))

    monkeypatch.setattr(flask_app, "get_ecg_model",
                        lambda: (FakeModel(), REAL_CARD))
    res = flask_app.predict_ecg_cnn((np.zeros((4, 280)), np.ones((4, 4))))
    assert res["prediction"] == "ABNORMAL"
    assert res["abnormal_type"] == "Ventricular ectopic beat"


def test_suppressed_classes_are_pooled_not_deleted(monkeypatch):
    """Their probability mass must still be visible, just unnamed."""
    import numpy as np

    class FakeModel:
        def predict(self, x, verbose=0):
            return np.tile(np.array([0.5, 0.2, 0.1, 0.1, 0.1]), (len(x), 1))

    monkeypatch.setattr(flask_app, "get_ecg_model",
                        lambda: (FakeModel(), REAL_CARD))
    dist = flask_app.predict_ecg_cnn((np.zeros((2, 280)), np.ones((2, 4))))["beat_distribution"]
    assert "Other / not reliably classified" in dist
    assert float(dist["Other / not reliably classified"]) == pytest.approx(40.0, abs=0.1)
    assert not any("Fusion" in k or "Unclassifiable" in k for k in dist)
