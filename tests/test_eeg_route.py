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


@pytest.mark.parametrize("model_used,episodes,expected", [
    (True, 2, "ABNORMAL"), (True, 0, "NORMAL"), (False, 0, "NOT ASSESSED")])
def test_eeg_verdict(model_used, episodes, expected):
    assert flask_app.eeg_verdict(model_used, episodes)["label"] == expected


def test_eeg_page_and_pdf_show_the_verdict(client):
    data = dict(PATIENT, duration="20", eeg_file=(_trace_png(), "trace.png"))
    html = client.post("/analyze_eeg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "EEG Classification" in html
    assert any(v in html for v in ("ABNORMAL", "NORMAL", "NOT ASSESSED"))
    fields = dict(PATIENT, eeg_json='{"model_used": true, "episodes_n": 0}')
    pdf = client.post("/download_eeg_report", data=fields).get_data()
    assert pdf.startswith(b"%PDF-")


def test_unassessed_report_never_says_normal(client):
    import base64
    import re
    import zlib
    pdf = client.post("/download_eeg_report",
                      data=dict(PATIENT, eeg_json='{"model_used": false}')).get_data()
    # ReportLab streams are ASCII85 wrapped around Flate.
    text = b"".join(zlib.decompress(base64.a85decode(m.strip(), adobe=True))
                    for m in re.findall(rb"stream\r?\n(.*?)endstream", pdf, re.S))
    assert b"NOT ASSESSED" in text
    assert b"(NORMAL" not in text


def _pdf_text(pdf):
    import base64
    import re
    import zlib
    return b"".join(zlib.decompress(base64.a85decode(m.strip(), adobe=True))
                    for m in re.findall(rb"stream\r?\n(.*?)endstream", pdf, re.S))


def test_eeg_pdf_reads_like_a_clinical_report(client):
    payload = ('{"model_used": true, "episodes_n": 1, "task_key": "seizure", '
               '"episodes": [{"onset_s": 2, "offset_s": 8, "duration_s": 6, '
               '"peak_score": 0.9, "spike_rate_per_s": 1.2}]}')
    pdf = client.post("/download_eeg_report",
                      data=dict(PATIENT, eeg_json=payload)).get_data()
    text = _pdf_text(pdf)
    for heading in (b"Clinical Impression", b"Recommendations",
                    b"Precautions", b"Reporting Clinician", b"(ABNORMAL"):
        assert heading in text, heading


def test_eeg_page_shows_the_same_advice(client):
    data = dict(PATIENT, duration="20", eeg_file=(_trace_png(), "trace.png"))
    html = client.post("/analyze_eeg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)
    for heading in ("Clinical Impression", "Recommendations", "Precautions"):
        assert heading in html, heading
