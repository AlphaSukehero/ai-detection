import pytest
from mlkit.registry import save_model_card, load_card, validate_card, RegistryError


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "m.keras"; p.write_bytes(b"x")
    save_model_card(str(p), "mri", ["a", "b"], [8, 8, 3], "rescale_255", {"test_accuracy": 0.9})
    card = load_card(str(p))
    assert card["task"] == "mri"
    assert card["classes"] == ["a", "b"]
    assert card["preprocessing"] == "rescale_255"


def test_validate_rejects_wrong_task():
    card = {"task": "mri", "classes": ["a"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="task"):
        validate_card(card, "satellite", ["a"], [8, 8, 3], "rescale_255")


def test_validate_rejects_class_mismatch():
    card = {"task": "mri", "classes": ["a", "b"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="class"):
        validate_card(card, "mri", ["a", "b", "c"], [8, 8, 3], "rescale_255")


def test_validate_rejects_preprocessing_mismatch():
    card = {"task": "mri", "classes": ["a"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="preprocessing"):
        validate_card(card, "mri", ["a"], [8, 8, 3], "vgg16_preprocess_input")


def test_missing_card_refuses_load(tmp_path):
    p = tmp_path / "undocumented.keras"; p.write_bytes(b"x")
    with pytest.raises(RegistryError, match="No model card"):
        load_card(str(p))
