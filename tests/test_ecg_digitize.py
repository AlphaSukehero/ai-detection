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


from ecg.digitize import detect_layout, extract_leads, LEAD_NAMES_12
from PIL import Image
import os


def test_single_strip_detected_as_single():
    img = np.full((120, 1200), 255.0)
    img[60, :] = 0.0
    assert detect_layout(img) == "single"


def test_wide_grid_of_panels_detected_as_twelve_lead():
    """Three rows of traces across a roughly page-shaped image."""
    img = np.full((900, 1200), 255.0)
    for row in range(3):
        y = 150 + row * 300
        img[y, :] = 0.0
    assert detect_layout(img) == "twelve_lead"


def test_panel_positions_map_to_correct_lead_names():
    """Pin the physical sheet layout: a set-equality check cannot catch a
    transposition, and Task 10 reads aVF from a specific panel."""
    from ecg.digitize import LEAD_PANELS_3X4
    assert LEAD_PANELS_3X4[0] == ["I", "aVR", "V1", "V4"]
    assert LEAD_PANELS_3X4[1] == ["II", "aVL", "V2", "V5"]
    assert LEAD_PANELS_3X4[2] == ["III", "aVF", "V3", "V6"]
    # aVF drives the axis calculation; assert its position explicitly.
    assert LEAD_PANELS_3X4[2][1] == "aVF"
    assert LEAD_PANELS_3X4[0][0] == "I"


def test_blank_panel_yields_no_signal():
    """A traceless panel must report nothing, not a fabricated flat line."""
    from ecg.digitize import _trace_row_band
    assert _trace_row_band(np.zeros((100, 300))) is None


def test_extract_leads_returns_named_signals(tmp_path):
    img = np.full((900, 1200), 255) .astype(np.uint8)
    for row in range(3):
        img[150 + row * 300, :] = 0
    path = os.path.join(tmp_path, "ecg.png")
    Image.fromarray(img).save(path)
    out = extract_leads(path)
    assert out["layout"] in {"single", "twelve_lead"}
    if out["layout"] == "twelve_lead":
        assert set(out["leads"]) == set(LEAD_NAMES_12)
