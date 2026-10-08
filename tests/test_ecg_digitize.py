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


def test_render_then_digitize_preserves_signal_shape(tmp_path):
    """Spec gate: round-trip correlation must exceed 0.95."""
    fs, n = 360.0, 1440
    t = np.arange(n) / fs
    truth = np.sin(2 * np.pi * 1.2 * t)

    h, w = 300, n
    img = np.full((h, w), 255.0)
    mid, amp = h // 2, h // 3
    for x in range(w):
        y = int(mid - truth[x] * amp)
        img[max(0, y - 1):min(h, y + 2), x] = 0.0

    path = os.path.join(tmp_path, "sine.png")
    Image.fromarray(img.astype(np.uint8)).save(path)

    got = extract_leads(path)["leads"]["II"]
    m = min(len(got), len(truth))
    r = np.corrcoef(got[:m], truth[:m])[0, 1]
    assert r > 0.95, f"round-trip correlation {r:.3f} below 0.95"


from ecg.parameters import analyse


def _render_strip(path, px_per_mm=10, hr_bpm=60.0, n_beats=8,
                  h=140, grid_gray=252, r_amp_px=40, thickness=3):
    """Render a synthetic rhythm strip at a KNOWN scale and a KNOWN rate.

    At 25 mm/s one pixel column is one sample, so the beat spacing in pixels
    is px_per_mm * 25 * 60 / hr_bpm. Vertical rulings every px_per_mm pixels
    give detect_grid_scale something real to find. The trace is drawn at a
    constant thickness in every column so it adds a constant to the column
    ink profile and cannot masquerade as grid periodicity.
    """
    fs = px_per_mm * 25.0
    rr_px = int(round(fs * 60.0 / hr_bpm))
    w = rr_px * n_beats
    img = np.full((h, w), 255.0)
    # Rulings 3 px wide: a 1-px comb has harmonics as strong as its
    # fundamental, and detect_grid_scale would lock onto px_per_mm / 2.
    for k in range(0, w, px_per_mm):
        img[:, k:k + 3] = float(grid_gray)          # vertical rulings

    signal = np.zeros(w)
    qrs_half = max(2, int(0.04 * fs))
    for b in range(n_beats):
        r = b * rr_px + rr_px // 2
        for off in range(-qrs_half, qrs_half + 1):
            i = r + off
            if 0 <= i < w:
                signal[i] = 1.0 - abs(off) / (qrs_half + 1.0)

    mid = h // 2
    for x in range(w):
        y = int(round(mid - signal[x] * r_amp_px))
        for t in range(thickness):
            yy = y + t - thickness // 2
            if 0 <= yy < h:
                img[yy, x] = 0.0

    Image.fromarray(img.astype(np.uint8)).save(path)
    return {"px_per_mm": px_per_mm, "fs": fs, "hr_bpm": hr_bpm}


def test_digitize_to_analyse_reports_the_true_heart_rate(tmp_path):
    """End-to-end calibration gate: image in, correct bpm out.

    This is the check that was missing while the image path passed the
    MIT-BIH fs=360 Hz to analyse(). A digitized trace has one sample per
    pixel COLUMN, so its real rate is px_per_mm * 25 mm/s; at 10 px/mm that
    is 250 Hz, and assuming 360 Hz reports a 60 bpm strip as 86.4 bpm.
    """
    path = os.path.join(tmp_path, "calibrated.png")
    truth = _render_strip(path, px_per_mm=10, hr_bpm=60.0)

    out = extract_leads(path)
    assert out["px_per_mm"] is not None, "grid not detected in the fixture"
    assert abs(out["px_per_mm"] - truth["px_per_mm"]) < 1.0, out["px_per_mm"]

    # fs is deliberately left at the CSV-path default: analyse() must derive
    # the real timebase from px_per_mm itself, so no caller can bypass it.
    report = analyse(out["leads"], fs=360.0,
                     px_per_mm=out["px_per_mm"], from_image=True)
    hr = report["heart_rate"].value
    assert hr is not None
    assert abs(hr - truth["hr_bpm"]) / truth["hr_bpm"] < 0.03, hr


def _render_chart_strip(path, hr_bpm=72.0, px_per_mm=16.0, h=600, w=1800):
    """A rhythm strip as a plotting library draws it: a black axes frame,
    light minor and heavier major rulings inside it, and dark title / tick
    label text in the margins outside it."""
    img = np.full((h, w), 255.0)
    top, bottom, left, right = 55, 512, 117, 1778
    minor, major = int(px_per_mm), int(px_per_mm) * 5
    for x in range(left, right, minor):
        img[top:bottom, x] = 215.0
    for y in range(top, bottom, minor):
        img[y, left:right] = 215.0
    for x in range(left, right, major):
        img[top:bottom, x] = 170.0
    for y in range(top, bottom, major):
        img[y, left:right] = 170.0

    fs = px_per_mm * 25.0
    rr_px = fs * 60.0 / hr_bpm
    base = 400
    y = np.full(w, float(base))
    for beat in np.arange(left + rr_px / 2, right - 20, rr_px):
        c = int(beat)
        for k, dy in enumerate([-30, -110, -170, -110, -30]):
            y[c + k * 3:c + k * 3 + 3] = base + dy
    y = y.astype(int)
    for x in range(left, right - 1):
        lo, hi = sorted((y[x], y[x + 1]))
        img[lo - 1:hi + 2, x] = 0.0

    img[top:top + 2, left:right] = 0.0                  # axes frame
    img[bottom:bottom + 2, left:right + 2] = 0.0
    img[top:bottom, left:left + 2] = 0.0
    img[top:bottom, right:right + 2] = 0.0
    rng = np.random.default_rng(0)
    for (r0, r1, c0, c1) in [(25, 42, 650, 1240),       # title
                             (555, 572, 860, 1030),     # x label
                             (90, 104, 60, 100), (160, 174, 60, 100),
                             (300, 314, 60, 100), (440, 454, 40, 100),
                             (528, 542, 330, 370), (528, 542, 1000, 1050)]:
        block = img[r0:r1, c0:c1]
        block[rng.random(block.shape) < 0.45] = 0.0
    Image.fromarray(img.astype(np.uint8)).save(path)
    return fs


def test_chart_style_strip_is_one_lead_not_a_twelve_lead_sheet(tmp_path):
    """The frame, title and heavy rulings are not trace bands. Counting them
    split a single strip into twelve 450-px panels."""
    path = os.path.join(tmp_path, "chart.png")
    _render_chart_strip(path)
    out = extract_leads(path)
    assert out["layout"] == "single"
    assert list(out["leads"]) == ["II"]


def test_chart_style_strip_reports_the_true_heart_rate(tmp_path):
    path = os.path.join(tmp_path, "chart.png")
    _render_chart_strip(path, hr_bpm=72.0)
    out = extract_leads(path)
    rep = analyse(out["leads"], px_per_mm=out["px_per_mm"], from_image=True)
    hr = rep["heart_rate"].value
    assert hr is not None and abs(hr - 72.0) < 4.0, hr
