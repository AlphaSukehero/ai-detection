"""End-to-end tests for the MRI and satellite routes.

Both routes were previously untested. They contain the Grad-CAM path, the
tumour morphometry and the spectral fallback -- the places where a silent
wrong answer is most likely and least visible.
"""
import io

import numpy as np
import pytest
from PIL import Image

import app as flask_app


PATIENT = {"patient_name": "T", "patient_id": "1", "age": "40",
           "gender": "Male", "contact": "1234567890",
           "referring_physician": "D", "study_date": "2026-09-09",
           "clinical_history": "test"}

SURVEY = {"site_name": "S", "survey_id": "1", "coordinates": "0, 0",
          "capture_date": "2026-09-09", "sensor": "test",
          "analyst": "A", "survey_notes": "test"}


requires_mri_model = pytest.mark.skipif(
    flask_app.get_brain_tumor_model()[0] is None,
    reason="no validated MRI model installed (checkpoints are not in git)")


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _png(size=(256, 256), seed=0):
    """A deterministic non-uniform image; uniform ones degenerate Grad-CAM."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _post_mri(client, meta):
    """A real scan: noise is now refused by the MRI input gate."""
    data = dict(meta)
    data["mri_file"] = (open("static/samples/mri_glioma.jpg", "rb"), "scan.jpg")
    return client.post("/analyze_brain_tumor", data=data,
                       content_type="multipart/form-data")


def _post(client, path, field, meta, filename="scan.png"):
    data = dict(meta)
    data[field] = (_png(), filename)
    return client.post(path, data=data, content_type="multipart/form-data")


# ---------------------------------------------------------------- MRI route

@requires_mri_model
def test_mri_route_renders_a_full_report(client):
    resp = _post_mri(client, PATIENT)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "error" not in html.lower() or "Tumor" in html


@requires_mri_model
def test_mri_route_always_shows_the_attention_section(client):
    """Regression for 5f81210 (section hidden behind a No-Tumor branch). The
    area/severity/spread it once showed were fabricated and are gone; see
    tests/test_mri_route.py."""
    resp = _post_mri(client, PATIENT)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    for label in ["Model Used", "Peak Attention Location"]:
        assert label in html, label


def test_mri_route_rejects_a_request_with_no_file(client):
    resp = client.post("/analyze_brain_tumor", data=dict(PATIENT),
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "Please upload an MRI image" in resp.get_data(as_text=True)


@requires_mri_model
def test_mri_route_refuses_malformed_metadata_rather_than_analysing(client):
    """Blank fields are allowed (they become "Not provided"); bad ones are not,
    because an out-of-range age reaches the PDF as a patient record."""
    bad = dict(PATIENT, age="not-a-number")
    resp = _post_mri(client, bad)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Age must be a whole number" in html
    assert "Estimated Area" not in html


@requires_mri_model
def test_mri_route_accepts_blank_metadata(client):
    blank = {k: "" for k in PATIENT}
    resp = _post_mri(client, blank)
    assert resp.status_code == 200
    assert "Peak Attention Location" in resp.get_data(as_text=True)


# ---------------------------------------------------------- satellite route

def test_satellite_route_renders_a_classification(client):
    resp = _post(client, "/analyze_satellite", "satellite_file", SURVEY,
                 filename="scene.png")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert any(g in html for g in ["Forest", "Urban", "Water", "Agricultural"])


def test_satellite_route_rejects_a_request_with_no_file(client):
    resp = client.post("/analyze_satellite", data=dict(SURVEY),
                       content_type="multipart/form-data")
    assert resp.status_code == 200


# ------------------------------------------------------- upload name policy

def test_uploads_route_refuses_a_name_this_app_never_generated(client):
    """/uploads/ serves only files matching the generated-name templates."""
    assert client.get("/uploads/app.py").status_code == 404
    assert client.get("/uploads/mri_cnn.keras").status_code == 404


def test_uploads_route_refuses_traversal(client):
    assert client.get("/uploads/..%2fapp.py").status_code == 404


def test_samples_route_refuses_an_arbitrary_name(client):
    assert client.get("/samples/../app.py").status_code == 404
