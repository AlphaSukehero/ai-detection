import numpy as np
from ecg.digitize import detect_grid_scale


def _grid_image(px_per_mm=5, w=600, h=400):
    """White page with dark rulings every px_per_mm pixels."""
    img = np.full((h, w), 255.0)
    img[:, ::px_per_mm] = 120.0
    img[::px_per_mm, :] = 120.0
    return img


def test_detects_known_grid_spacing():
    scale = detect_grid_scale(_grid_image(px_per_mm=5))
    assert scale is not None
    assert abs(scale - 5.0) < 0.6


def test_returns_none_for_blank_page():
    assert detect_grid_scale(np.full((400, 600), 255.0)) is None
