"""Comparison, trends and the complete patient-history PDF."""
import re

from tests.conftest_records import register, rclient  # noqa: F401
from tests.test_pdf_reports import _pdf_text
from tests.test_study_binding import _ecg, _study_ids


def _two_ecgs(client):
    pid = register(client)
    a = _study_ids(_ecg(client, pid, "2026-08-01", sample="normal"))[0]
    b = _study_ids(_ecg(client, pid, "2026-09-01", sample="abnormal"))[0]
    return pid, a, b


def test_compare_page_shows_both_studies_side_by_side(rclient):
    pid, a, b = _two_ecgs(rclient)
    html = rclient.get(f"/patients/{pid}/compare?a={a}&b={b}").get_data(as_text=True)
    assert a in html and b in html
    assert "Heart rate" in html and "Before" in html
    assert f"/patients/{pid}/studies/{a}/files/waveform" in html
    assert f"/patients/{pid}/studies/{b}/files/waveform" in html


def test_compare_orders_older_first_whatever_the_query(rclient):
    pid, a, b = _two_ecgs(rclient)
    html = rclient.get(f"/patients/{pid}/compare?a={b}&b={a}").get_data(as_text=True)
    section = html[html.index("Before:"):]
    assert section.index("2026-08-01") < section.index("2026-09-01")
    assert section.index(a) < section.index(b)


def test_compare_without_a_choice_offers_a_picker(rclient):
    pid, a, b = _two_ecgs(rclient)
    html = rclient.get(f"/patients/{pid}/compare").get_data(as_text=True)
    assert f'value="{a}"' in html and f'value="{b}"' in html


def test_compare_across_modalities_is_refused(rclient):
    from tests.test_eeg_route import _trace_png
    pid = register(rclient)
    a = _study_ids(_ecg(rclient, pid))[0]
    data = {"record_id": pid, "duration": "20", "task": "seizure",
            "eeg_file": (_trace_png(), "t.png")}
    b = _study_ids(rclient.post("/analyze_eeg", data=data,
                                content_type="multipart/form-data").get_data(as_text=True))[0]
    resp = rclient.get(f"/patients/{pid}/compare?a={a}&b={b}")
    assert resp.status_code == 400
    assert "same modality" in resp.get_data(as_text=True)


def test_trend_chart_is_a_png(rclient):
    pid, _a, _b = _two_ecgs(rclient)
    resp = rclient.get(f"/patients/{pid}/trends/ecg.png")
    assert resp.status_code == 200 and resp.mimetype == "image/png"
    assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_trend_for_a_modality_with_no_studies_is_404(rclient):
    pid, _a, _b = _two_ecgs(rclient)
    assert rclient.get(f"/patients/{pid}/trends/mri.png").status_code == 404


def test_patient_page_links_trends_and_comparison(rclient):
    pid, a, b = _two_ecgs(rclient)
    html = rclient.get(f"/patients/{pid}").get_data(as_text=True)
    assert f"/patients/{pid}/trends/ecg.png" in html
    assert f"/patients/{pid}/compare?a={a}&amp;b={b}" in html or \
        f"/patients/{pid}/compare?a={a}&b={b}" in html
    assert f"/patients/{pid}/history.pdf" in html


def test_history_pdf_contains_every_study_and_the_comparison(rclient):
    pid, a, b = _two_ecgs(rclient)
    resp = rclient.get(f"/patients/{pid}/history.pdf")
    assert resp.status_code == 200 and resp.mimetype == "application/pdf"
    text = _pdf_text(resp.data)
    for s in (b"COMPLETE PATIENT HISTORY", b"Asha Rao", pid.encode(),
              a.encode(), b.encode(), b"Visit Timeline", b"Contents",
              b"Latest vs previous", b"ECG ANALYSIS REPORT"):
        assert s in text, s
    pages = len(re.findall(rb"/Type /Page[^s]", resp.data))
    assert pages >= 4, pages


def test_history_pdf_for_an_unknown_patient_is_404(rclient):
    assert rclient.get("/patients/PT-999999/history.pdf").status_code == 404
