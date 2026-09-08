import os

import pytest

from mlkit.registry import load_card, validate_card

CASES = [
    ("model/mri_vgg16.keras", "mri", 4, [224, 224, 3], "vgg16_preprocess_input"),
    ("model/mri_cnn.keras", "mri", 4, [128, 128, 3], "rescale_255"),
    ("model/satellite_best.keras", "satellite", 10, [64, 64, 3], "rescale_255"),
    ("model/ecg_cnn.keras", "ecg", 5, [280, 1], "beat_zscore"),
]


@pytest.mark.parametrize("path,task,n,shape,prep", CASES)
def test_card_matches_expectations(path, task, n, shape, prep):
    if not os.path.exists(path):
        pytest.skip(f"{path} not trained yet")
    card = load_card(path)
    assert card["task"] == task
    assert len(card["classes"]) == n
    validate_card(card, task, card["classes"], shape, prep)


@pytest.mark.parametrize("path,task,n,shape,prep", CASES)
def test_model_output_matches_card(path, task, n, shape, prep):
    if not os.path.exists(path):
        pytest.skip(f"{path} not trained yet")
    from tensorflow.keras.models import load_model
    model = load_model(path, compile=False)
    assert model.output_shape[-1] == n, (
        f"{path} outputs {model.output_shape[-1]}, card says {n}")
