import app as flask_app


def _post(client, path, extra):
    data = {"patient_name": "T", "patient_id": "1", "age": "40",
            "gender": "Male", "contact": "1234567890",
            "referring_physician": "D", "study_date": "2026-09-09",
            "clinical_history": "test"}
    data.update(extra)
    return client.post(path, data=data, content_type="multipart/form-data")


def test_ecg_route_still_renders_successfully():
    """Task 13 replaces the parameter source; the template labels for Rhythm,
    ST Segment and QRS Axis arrive in Task 14, so they are asserted there."""
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        resp = _post(client, "/analyze_ecg", {"sample_type": "normal"})
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "ECG Waveform Parameters" in html


def test_pr_is_no_longer_the_hardcoded_constant():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        resp = _post(client, "/analyze_ecg", {"sample_type": "normal"})
        html = resp.get_data(as_text=True)
        assert "145.0 ms" not in html


def test_ecg_page_reports_new_parameters():
    """The three parameters the app never computed before must now appear."""
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        resp = _post(client, "/analyze_ecg", {"sample_type": "normal"})
        html = resp.get_data(as_text=True)
        assert resp.status_code == 200
        for label in ["Rhythm", "ST Segment", "QRS Axis"]:
            assert label in html, label
