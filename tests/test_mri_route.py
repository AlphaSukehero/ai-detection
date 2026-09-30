"""Brain-tumour page: correct loading, refusal of non-MRI input, and no
lesion metric the pipeline cannot actually measure.

Verified 2026-09-30: "area" was the share of Grad-CAM pixels above the 90th
percentile -- ~10% for every scan by construction -- so severity always read
"Moderate", and a flat heatmap gave 100% / "High" / "Extensive".
"""
import io
import re

import numpy as np
import pytest
from PIL import Image

import app as flask_app
from tests.test_pdf_reports import _pdf_text
from tests.test_routes import PATIENT, requires_mri_model

FABRICATED = ["Estimated Area", "Estimated Severity", "Estimated Spread",
              "Bounding Width", "Bounding Height"]


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _file(arr_or_img, fmt="PNG"):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def _analyse(client, buf, name="scan.png"):
    data = dict(PATIENT, mri_file=(buf, name))
    return client.post("/analyze_brain_tumor", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)


def _prediction(html):
    m = re.search(r'name="prediction" value="([^"]*)"', html)
    return m.group(1) if m else None


@requires_mri_model
def test_16bit_glioma_is_still_glioma(client):
    g = np.asarray(Image.open("static/samples/mri_glioma.jpg").convert("L"),
                   dtype=np.uint16) * 257
    assert _prediction(_analyse(client, _file(g))) == "Glioma"


def test_noise_is_refused_not_classified(client):
    rng = np.random.default_rng(0)
    html = _analyse(client, _file(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)))
    assert "looks like noise" in html
    assert _prediction(html) is None


@requires_mri_model
def test_page_shows_no_fabricated_lesion_metrics(client):
    html = _analyse(client, open("static/samples/mri_glioma.jpg", "rb"), "g.jpg")
    assert _prediction(html) == "Glioma"
    for label in FABRICATED:
        assert label not in html, label


@requires_mri_model
def test_page_names_the_model_that_actually_ran(client):
    html = _analyse(client, open("static/samples/mri_glioma.jpg", "rb"), "g.jpg")
    _model, card = flask_app.get_brain_tumor_model()
    assert card["_name"] in html


@requires_mri_model
def test_attention_map_failure_is_shown(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no conv layer")
    monkeypatch.setattr(flask_app, "generate_gradcam_heatmap", boom)
    html = _analyse(client, open("static/samples/mri_glioma.jpg", "rb"), "g.jpg")
    assert "Attention map unavailable" in html and "no conv layer" in html


def test_pdf_has_no_fabricated_lesion_metrics(client):
    data = dict(PATIENT, prediction="Glioma", confidence="91.0",
                location="Left frontal", area="9.99", severity="Moderate",
                spread="Moderate", width="87", height="86")
    pdf = client.post("/download_brain_tumor_report", data=data).data
    assert pdf[:5] == b"%PDF-"
    text = _pdf_text(pdf)
    assert b"MRI Classification" in text, "extraction must see real text"
    for s in (b"Area of Activation", b"Severity Indicator", b"Spread Indicator",
              b"Bounding Width"):
        assert s not in text, s
