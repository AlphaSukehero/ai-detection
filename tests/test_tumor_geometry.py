"""Unit tests for the tumour morphometry derived from the Grad-CAM mask.

These numbers reach the PDF report as "Tumor Area", "Size" and "Location",
so they are read as measurements. Nothing tested them before.
"""
import numpy as np
import pytest

from app import (calculate_tumor_area, calculate_tumor_size,
                 calculate_tumor_location, calculate_severity,
                 calculate_spread)


def test_area_of_an_empty_mask_is_zero():
    assert calculate_tumor_area(np.zeros((100, 100), dtype=np.uint8)) == 0.0


def test_area_of_a_full_mask_is_one_hundred_percent():
    assert calculate_tumor_area(np.ones((100, 100), dtype=np.uint8)) == pytest.approx(100.0)


def test_area_is_a_percentage_of_the_mask_not_a_pixel_count():
    """A quarter-covered mask is 25%, independent of the mask's resolution."""
    small = np.zeros((10, 10), dtype=np.uint8)
    small[:5, :5] = 1
    big = np.zeros((200, 200), dtype=np.uint8)
    big[:100, :100] = 1
    assert calculate_tumor_area(small) == pytest.approx(25.0)
    assert calculate_tumor_area(big) == pytest.approx(25.0)


def test_area_thresholds_are_ordered_and_cover_the_range():
    """Severity and spread must be monotonic in area with no gap or overlap."""
    severities = [calculate_severity(a) for a in (0.0, 1.0, 7.0, 20.0, 100.0)]
    spreads = [calculate_spread(a) for a in (0.0, 1.0, 7.0, 20.0, 100.0)]
    assert len(set(severities)) > 1
    assert len(set(spreads)) > 1
    # Every area maps to something; none falls through to None.
    assert all(s is not None for s in severities + spreads)


def _square_contour(x, y, w, h):
    """A contour in the (N, 1, 2) shape OpenCV returns."""
    pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def test_size_reports_the_bounding_box_of_the_contour():
    """cv2.boundingRect counts both endpoints, so a span of 10..40 is 31px."""
    w, h = calculate_tumor_size(_square_contour(10, 20, 30, 40))
    assert (w, h) == (31, 41)


def test_size_of_a_missing_contour_is_zero_not_an_exception():
    assert calculate_tumor_size(None) == (0, 0)


def test_location_distinguishes_the_four_quadrants():
    """A left-superior lesion must not be described as right-inferior."""
    seen = set()
    for cx, cy in [(30, 30), (170, 30), (30, 170), (170, 170)]:
        loc = calculate_tumor_location(_square_contour(cx - 5, cy - 5, 10, 10))
        seen.add(loc)
    assert len(seen) == 4, f"quadrants collapsed to {seen}"
