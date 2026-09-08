"""Model cards: what a model was trained on and how its inputs are prepared.

Two severe defects in this codebase were undetectable because nothing recorded
these facts: a classifier fed /255 inputs when it was trained with vgg16
preprocess_input (it answered "No Tumor" for every scan), and a brain-tumor
model serving as a land-cover classifier (it answered one class for every
scene). A model card plus validate_card() turns both into a refused load.
"""
import json
import os
from datetime import date


class RegistryError(RuntimeError):
    """A model does not match what the calling code expects."""


def card_path(model_path):
    return os.path.splitext(model_path)[0] + ".json"


def save_model_card(model_path, task, classes, input_shape, preprocessing, metrics):
    """Write the sidecar describing a freshly trained model."""
    card = {
        "task": task,
        "classes": list(classes),
        "input_shape": list(input_shape),
        "preprocessing": preprocessing,
        "metrics": dict(metrics),
        "trained": date.today().isoformat(),
    }
    with open(card_path(model_path), "w") as f:
        json.dump(card, f, indent=2)
    return card


def load_card(model_path):
    path = card_path(model_path)
    if not os.path.exists(path):
        raise RegistryError(
            f"No model card beside {model_path}; refusing to load an undocumented model."
        )
    with open(path) as f:
        return json.load(f)


def validate_card(card, task, classes, input_shape, preprocessing):
    """Raise RegistryError unless the card matches the caller's expectations."""
    if card.get("task") != task:
        raise RegistryError(
            f"task mismatch: card says {card.get('task')!r}, caller expects {task!r}"
        )
    if list(card.get("classes", [])) != list(classes):
        raise RegistryError(
            f"class mismatch: card has {card.get('classes')}, caller expects {list(classes)}"
        )
    if list(card.get("input_shape", [])) != list(input_shape):
        raise RegistryError(
            f"input_shape mismatch: card {card.get('input_shape')} vs {list(input_shape)}"
        )
    if card.get("preprocessing") != preprocessing:
        raise RegistryError(
            f"preprocessing mismatch: card {card.get('preprocessing')!r} vs {preprocessing!r}"
        )
