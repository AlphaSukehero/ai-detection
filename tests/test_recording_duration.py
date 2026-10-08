"""The recording duration is entered once, with the patient details, and is
honoured for every input kind -- but never against the evidence in the file.

Before this the duration sat in the upload panel marked "images only": a CSV
was always read at 360 Hz, a time column was read as the ECG, and a duration
that contradicted the printed grid was believed over the grid.
"""
import io
import re

import numpy as np
import pytest
from PIL import Image

import app as flask_app
from ecg.timebase import table_signal
from tests.conftest_records import register, rclient  # noqa: F401
from tests.test_ecg_digitize import _render_chart_strip
from tests.test_ecg_route import _post
from tests.test_ecg_timebase import _gridless_ecg_png, _heart_rate
from tests.test_routes import PATIENT


def _ecg(fs=360.0, seconds=10.0, bpm=72.0):
    """A clean synthetic lead: P, QRS and T on a flat baseline."""
    t = np.arange(0, seconds, 1.0 / fs)
    x = np.zeros_like(t)
    for beat in np.arange(0.5, seconds, 60.0 / bpm):
        x += 0.12 * np.exp(-((t - beat + 0.16) ** 2) / (2 * 0.02 ** 2))
        x += 1.0 * np.exp(-((t - beat) ** 2) / (2 * 0.010 ** 2))
        x += 0.25 * np.exp(-((t - beat - 0.25) ** 2) / (2 * 0.04 ** 2))
    return t, x


def _csv(*columns, header=None):
    rows = [",".join(f"{v:.6f}" for v in r) for r in zip(*columns)]
    text = "\n".join(([header] if header else []) + rows) + "\n"
    return io.BytesIO(text.encode())


def _analyse(upload, name, **extra):
    with flask_app.app.test_client() as c:
        return _post(c, "/analyze_ecg", dict(
            extra, ecg_file=(upload, name))).get_data(as_text=True)


def _bpm(html):
    hr = _heart_rate(html)
    assert hr and hr != "—", hr
    return float(hr.split()[0])


# ------------------------------------------------------------ the form

@pytest.mark.parametrize("path, field, upload_heading", [
    ("/ecg", "ecg_duration", "ECG Signal Data"),
    ("/eeg", "duration", "EEG Recording</h3>"),
])
def test_duration_is_asked_with_the_patient_details(path, field, upload_heading):
    with flask_app.app.test_client() as c:
        html = c.get(path).get_data(as_text=True)
    assert html.count(f'name="{field}"') == 1
    assert "Patient Details" in html
    assert (html.index("Patient Details") < html.index(f'name="{field}"')
            < html.index(upload_heading))
    assert "(seconds) — images only" not in html


def test_mri_page_does_not_ask_for_a_duration():
    with flask_app.app.test_client() as c:
        html = c.get("/brain-tumor").get_data(as_text=True)
    assert "Recording Duration" not in html


def test_duration_is_asked_for_a_registered_patient_too(rclient):
    pid = register(rclient)
    html = rclient.get(f"/ecg?record_id={pid}").get_data(as_text=True)
    assert "Study for a registered patient" in html
    assert html.count('name="ecg_duration"') == 1


# ------------------------------------------------------------ ECG data files

def test_plain_column_is_read_at_the_documented_default():
    _t, x = _ecg()
    signal, fs, source = table_signal(x.reshape(-1, 1), None)
    assert fs == 360.0 and len(signal) == len(x)
    assert "360 Hz assumed" in source


def test_a_time_column_is_the_timebase_not_the_signal():
    t, x = _ecg(fs=250.0)
    signal, fs, source = table_signal(np.column_stack([t, x]), None)
    assert fs == pytest.approx(250.0, rel=0.01)
    assert np.allclose(signal, x)
    assert "time column" in source


def test_a_millisecond_time_column_is_recognised():
    t, x = _ecg(fs=500.0)
    _signal, fs, _source = table_signal(np.column_stack([t * 1000.0, x]), None)
    assert fs == pytest.approx(500.0, rel=0.01)


def test_a_sample_index_column_is_dropped_without_setting_the_rate():
    _t, x = _ecg()
    signal, fs, _source = table_signal(
        np.column_stack([np.arange(len(x)), x]), None)
    assert fs == 360.0
    assert np.allclose(signal, x)


def test_stated_duration_sets_the_sampling_rate():
    _t, x = _ecg(fs=250.0)
    _signal, fs, source = table_signal(x.reshape(-1, 1), 10.0)
    assert fs == pytest.approx(250.0)
    assert "stated duration 10 s" in source


def test_an_impossible_duration_is_refused_with_the_reason():
    _t, x = _ecg()
    with pytest.raises(ValueError, match="samples in 600 s"):
        table_signal(x.reshape(-1, 1), 600.0)


def test_csv_with_a_time_column_reports_the_true_heart_rate():
    t, x = _ecg(fs=360.0, bpm=72.0)
    html = _analyse(_csv(t, x, header="time,ecg"), "rec.csv")
    assert abs(_bpm(html) - 72) < 4


def test_csv_at_another_rate_reports_the_true_heart_rate_with_a_duration():
    _t, x = _ecg(fs=250.0, bpm=72.0)
    html = _analyse(_csv(x), "rec.csv", ecg_duration="10")
    assert abs(_bpm(html) - 72) < 4
    assert "250 Hz" in html


# ------------------------------------------------------------ ECG images

def test_duration_contradicting_the_printed_grid_does_not_override_it(tmp_path):
    """The strip shows about four seconds; ten was entered. The grid is
    measured from the page, so it wins, and the page says so."""
    path = str(tmp_path / "strip.png")
    _render_chart_strip(path, hr_bpm=72.0)
    with open(path, "rb") as fh:
        html = _analyse(io.BytesIO(fh.read()), "strip.png", ecg_duration="10")
    assert abs(_bpm(html) - 72) < 6
    assert "does not match the printed grid" in html


def test_duration_agreeing_with_the_printed_grid_is_used(tmp_path):
    path = str(tmp_path / "strip.png")
    _render_chart_strip(path, hr_bpm=72.0)
    with open(path, "rb") as fh:
        html = _analyse(io.BytesIO(fh.read()), "strip.png", ecg_duration="4.15")
    assert abs(_bpm(html) - 72) < 6
    assert "does not match the printed grid" not in html
    assert "stated duration 4.15 s" in html


def test_blank_margins_are_not_part_of_a_gridless_strip():
    """A scan has white paper either side of the trace; the duration covers
    the trace, not the page."""
    strip = np.asarray(Image.open(_gridless_ecg_png(bpm=72, duration=10.0)))
    page = np.full((strip.shape[0], strip.shape[1] + 1000), 255, np.uint8)
    page[:, 500:500 + strip.shape[1]] = strip
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, format="PNG")
    buf.seek(0)
    html = _analyse(buf, "scan.png", ecg_duration="10")
    assert abs(_bpm(html) - 72) < 6


# ------------------------------------------------------------ EEG

def test_eeg_csv_takes_its_rate_from_the_duration():
    from tests.test_eeg_csv import _csv_upload
    with flask_app.app.test_client() as c:
        data = dict(PATIENT, duration="20", task="seizure",
                    eeg_file=(_csv_upload(seconds=20.0), "rec.csv"))
        html = c.post("/analyze_eeg", data=data,
                      content_type="multipart/form-data").get_data(as_text=True)
    assert "Per-Window Clinical Parameters" in html
    assert "CSV, 2 channels at 128 Hz" in html
