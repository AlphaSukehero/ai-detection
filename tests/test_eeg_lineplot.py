"""Line-plot representation: renderable, but no model serves it until one is
trained and validated for it."""
import io

import numpy as np
from PIL import Image

import app as flask_app
from eeg.images import lineplot, render_batch
from tests.test_routes import PATIENT

FS = 128.0


def _sine(freq, seconds=2.0):
    t = np.arange(0, seconds, 1 / FS)
    return 20 * np.sin(2 * np.pi * freq * t)


def test_lineplot_is_a_normalised_trace_image():
    img = lineplot(_sine(5.0), FS)
    assert img.shape == (64, 64) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    # Every time column carries ink: the trace is continuous.
    assert np.all(img.max(axis=0) > 0.5)


def test_lineplot_distinguishes_rhythms():
    assert not np.allclose(lineplot(_sine(3.0), FS), lineplot(_sine(12.0), FS))


def test_lineplot_is_gain_invariant():
    assert np.allclose(lineplot(_sine(5.0), FS), lineplot(10 * _sine(5.0), FS))


def test_render_batch_accepts_lineplot():
    batch = render_batch(np.stack([_sine(5.0)] * 3), FS, kind="lineplot")
    assert batch.shape == (3, 64, 64, 1)


def test_no_lineplot_model_means_not_assessed():
    model, card = flask_app.get_eeg_model("seizure", "lineplot")
    assert model is None and card is None


def test_route_lineplot_reports_not_assessed_but_measures(tmp_path):
    from tests.test_eeg_route import _trace_png
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        data = dict(PATIENT, duration="20", task="seizure", representation="lineplot",
                    eeg_file=(_trace_png(), "trace.png"))
        html = c.post("/analyze_eeg", data=data,
                      content_type="multipart/form-data").get_data(as_text=True)
    assert "NOT ASSESSED" in html
    assert "No validated line-plot model" in html
    assert "Per-Window Clinical Parameters" in html


def test_form_offers_the_representation_choice():
    with flask_app.app.test_client() as c:
        html = c.get("/eeg").get_data(as_text=True)
    assert 'name="representation"' in html and 'value="lineplot"' in html
