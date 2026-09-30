"""An ECG image with no printed grid needs a stated timebase.

Before this, a gridless image silently ran beat detection at the MIT-BIH
360 Hz and reported every interval unavailable, with no way for the user to
supply the recording duration.
"""
import io
import re

import numpy as np
from PIL import Image

import app as flask_app
from tests.test_ecg_route import _post


def _gridless_ecg_png(bpm=72, duration=10.0, width=2500, height=300):
    """White image, black trace: flat baseline with a sharp QRS per beat."""
    arr = np.full((height, width), 255, np.uint8)
    base = height * 2 // 3
    px_per_s = width / duration
    y = np.full(width, base, float)
    for beat in np.arange(0.4, duration, 60.0 / bpm):
        c = int(beat * px_per_s)
        for k, dy in enumerate([-20, -120, -30, 20, 0]):
            if 0 <= c + k * 2 < width:
                y[c + k * 2:c + k * 2 + 2] = base + dy
    y = y.astype(int)
    for x in range(width - 1):
        lo, hi = sorted((y[x], y[x + 1]))
        arr[lo - 1:hi + 2, x] = 0
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _heart_rate(html):
    m = re.search(r'name="heart_rate" value="([^"]*)"', html)
    return m.group(1) if m else None


def test_ecg_form_asks_for_duration_and_paper_speed():
    with flask_app.app.test_client() as c:
        html = c.get("/ecg").get_data(as_text=True)
    assert 'name="ecg_duration"' in html
    assert 'name="paper_speed"' in html


def test_gridless_image_with_stated_duration_is_measured():
    with flask_app.app.test_client() as c:
        html = _post(c, "/analyze_ecg", {
            "ecg_file": (_gridless_ecg_png(), "strip.png"),
            "ecg_duration": "10", "paper_speed": "25"}).get_data(as_text=True)
    hr = _heart_rate(html)
    assert hr and hr != "—", hr
    assert abs(float(hr.split()[0]) - 72) < 6


def test_gridless_image_without_duration_says_why_nothing_was_measured():
    with flask_app.app.test_client() as c:
        html = _post(c, "/analyze_ecg", {
            "ecg_file": (_gridless_ecg_png(), "strip.png")}).get_data(as_text=True)
    assert _heart_rate(html) == "—"
    assert "No timebase" in html
