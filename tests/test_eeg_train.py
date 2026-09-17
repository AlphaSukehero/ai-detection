import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tensorflow as tf  # noqa: E402

from train_eeg import build, shuffled  # noqa: E402


def test_model_round_trips_through_default_safe_load(tmp_path):
    # The dashboard loads with Keras defaults (safe_mode=True), which refuses
    # Python lambdas. A model that only loads with safe_mode=False is a model
    # the app cannot use.
    model = build((32, 64, 1))
    x = np.random.default_rng(0).random((2, 32, 64, 1)).astype("float32")
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(loaded.predict(x, verbose=0),
                               model.predict(x, verbose=0), rtol=1e-5)


def test_shuffle_spreads_positives_into_validation_tail():
    # validation_split takes the LAST fraction unshuffled; source-ordered data
    # would leave it with no positives.
    y = np.array([1] * 100 + [0] * 900)
    x = np.arange(1000)
    xs, ys = shuffled(x, y, seed=42)
    assert ys[-150:].sum() > 0
    assert sorted(xs.tolist()) == x.tolist()
    np.testing.assert_array_equal(y[xs], ys)
