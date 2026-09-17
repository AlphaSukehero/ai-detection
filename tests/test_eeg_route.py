"""The EEG module is served by the main Flask site, not a separate app."""
import io

import numpy as np
import pytest
from PIL import Image

import app as flask_app
from tests.test_routes import PATIENT


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _trace_png(freq=5.0, duration=20.0, width=2000, height=200):
    t = np.linspace(0, duration, width)
    y = (height / 2 - 0.35 * height * np.sin(2 * np.pi * freq * t)).astype(int)
    arr = np.full((height, width), 255, np.uint8)
    for x in range(width):
        arr[max(y[x] - 1, 0):y[x] + 2, x] = 0
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.mark.parametrize("path", ["/", "/ecg", "/eeg", "/brain-tumor", "/satellite"])
def test_every_page_links_every_module(client, path):
    html = client.get(path).get_data(as_text=True)
    for href in ['href="/ecg"', 'href="/eeg"', 'href="/brain-tumor"', 'href="/satellite"']:
        assert href in html, (path, href)


def test_eeg_rejects_a_request_with_no_file(client):
    resp = client.post("/analyze_eeg", data=dict(PATIENT),
                       content_type="multipart/form-data")
    assert "Please upload an EEG recording" in resp.get_data(as_text=True)


def test_eeg_image_without_timebase_is_refused(client):
    data = dict(PATIENT, eeg_file=(_trace_png(), "trace.png"))
    html = client.post("/analyze_eeg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "Analysis Error" in html and "Analysis Result" not in html


def test_eeg_image_with_duration_renders_parameters(client):
    data = dict(PATIENT, duration="20", task="seizure",
                eeg_file=(_trace_png(), "trace.png"))
    resp = client.post("/analyze_eeg", data=data, content_type="multipart/form-data")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Analysis Result" in html, html[html.find("Analysis Error"):][:300]
    assert "Per-Window Clinical Parameters" in html
    assert "eeg_timeline_" in html


def test_eeg_result_page_offers_a_pdf_that_downloads(client):
    import re
    from html import unescape
    data = dict(PATIENT, duration="20", eeg_file=(_trace_png(), "trace.png"))
    html = client.post("/analyze_eeg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)
    form = html[html.index('action="/download_eeg_report"'):]
    form = form[:form.index("</form>")]
    fields = {k: unescape(v1 or v2) for k, v1, v2 in
              re.findall(r'name="([^"]+)" value=(?:"([^"]*)"|\'([^\']*)\')', form)}
    # The payload must survive the HTML attribute intact; a PDF built from a
    # truncated one still downloads, just with every finding "Not provided".
    import json
    payload = json.loads(fields["eeg_json"])
    assert payload["n_windows"] > 0 and payload["band_means"]
    resp = client.post("/download_eeg_report", data=fields)
    assert resp.status_code == 200
    assert resp.get_data().startswith(b"%PDF-")
    assert "EEG_Report_" in resp.headers["Content-Disposition"]


def test_eeg_report_survives_a_malformed_payload(client):
    resp = client.post("/download_eeg_report", data=dict(PATIENT, eeg_json="{bad"))
    assert resp.status_code == 200
    assert resp.get_data().startswith(b"%PDF-")
