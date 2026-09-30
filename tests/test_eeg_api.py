"""POST /api/eeg/analyze: the same analysis as the page, as JSON."""
from tests.conftest_records import register, rclient  # noqa: F401
from tests.test_eeg_csv import _csv_upload
from tests.test_eeg_route import _trace_png
from tests.test_routes import PATIENT


def _post(client, **extra):
    data = dict(PATIENT, task="seizure", duration="20",
                eeg_file=(_trace_png(), "trace.png"))
    data.update(extra)
    return client.post("/api/eeg/analyze", data=data,
                       content_type="multipart/form-data")


def test_api_returns_the_full_window_array(rclient):
    resp = _post(rclient)
    assert resp.status_code == 200 and resp.is_json
    j = resp.get_json()
    assert j["ok"] is True
    assert j["verdict"]["label"] in ("NORMAL", "ABNORMAL", "NOT ASSESSED")
    assert j["n_windows"] == len(j["windows"]) > 5
    w = j["windows"][0]
    assert {"start_s", "stop_s", "score", "artifact", "rsp", "spikes"} <= set(w)
    assert set(w["rsp"]) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert j["timeline_url"].startswith("/uploads/eeg_timeline_")
    assert j["representation"] == "scalogram"
    assert "peak" in j and "episodes" in j and j["notes"]["impression"]


def test_api_report_json_downloads_the_same_pdf(rclient):
    j = _post(rclient).get_json()
    pdf = rclient.post("/download_eeg_report", data=j["report_form"])
    assert pdf.data[:5] == b"%PDF-"


def test_api_errors_are_json_with_400(rclient):
    resp = rclient.post("/api/eeg/analyze", data=dict(PATIENT),
                        content_type="multipart/form-data")
    assert resp.status_code == 400 and resp.get_json() == {
        "ok": False, "error": "Please upload an EEG recording or trace image."}


def test_api_csv_needs_a_sampling_rate(rclient):
    resp = _post(rclient, eeg_file=(_csv_upload(), "r.csv"), duration="")
    assert resp.status_code == 400
    assert "sampling rate" in resp.get_json()["error"]
    ok = _post(rclient, eeg_file=(_csv_upload(), "r.csv"), sampling_rate="128")
    assert ok.status_code == 200 and "CSV" in ok.get_json()["provenance"]


def test_api_lineplot_is_not_assessed(rclient):
    j = _post(rclient, representation="lineplot").get_json()
    assert j["verdict"]["label"] == "NOT ASSESSED"
    assert "line-plot" in j["verdict"]["detail"]
    assert all(w["score"] is None for w in j["windows"])


def test_api_saves_to_the_patient_record(rclient):
    pid = register(rclient)
    j = _post(rclient, record_id=pid).get_json()
    assert j["saved"]["study_id"].startswith("ST-")
    assert j["saved"]["url"] == f"/patients/{pid}/studies/{j['saved']['study_id']}"
    assert rclient.get(j["saved"]["pdf_url"]).data[:5] == b"%PDF-"


def test_eeg_page_loads_the_script_and_keeps_the_plain_form():
    import app as flask_app
    with flask_app.app.test_client() as c:
        html = c.get("/eeg").get_data(as_text=True)
    assert 'src="/static/eeg.js"' in html
    assert 'action="/analyze_eeg"' in html     # still works without JS
    assert c.get("/static/eeg.js").status_code == 200
