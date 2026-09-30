"""What reaches the MRI classifier must be a greyscale brain scan, loaded right.

Verified 2026-09-30: a 16-bit copy of each tumour sample was clipped to white
by Image.convert("RGB") and classified "No Tumor 100%"; random noise and
20x20 thumbnails received confident verdicts.
"""
import glob
import io

import numpy as np
import pytest
from PIL import Image

from vision.imageio import check_mri_input, load_mri_rgb

SAMPLES = sorted(glob.glob("static/samples/mri_*.jpg"))


def _to_16bit(path, tmp_path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.uint16) * 257
    out = tmp_path / "scan16.png"
    Image.fromarray(g).save(out)
    assert Image.open(out).mode.startswith("I")
    return out


@pytest.mark.parametrize("path", SAMPLES)
def test_16bit_scan_loads_like_its_8bit_original(path, tmp_path):
    ref = np.asarray(load_mri_rgb(path).convert("L"), float)
    got = np.asarray(load_mri_rgb(_to_16bit(path, tmp_path)).convert("L"), float)
    assert got.std() > 10, "16-bit scan collapsed to a flat image"
    assert np.corrcoef(ref.ravel(), got.ravel())[0, 1] > 0.95


@pytest.mark.parametrize("path", SAMPLES)
def test_real_scans_pass_the_input_check(path):
    assert check_mri_input(load_mri_rgb(path)) is None


def test_noise_is_refused():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    assert "noise" in check_mri_input(img)


def test_a_thumbnail_is_refused():
    img = load_mri_rgb(SAMPLES[0]).resize((20, 20))
    assert "too small" in check_mri_input(img)


def test_a_colour_photo_is_refused():
    a = np.zeros((256, 256, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, 256)[None, :]
    a[..., 1] = np.linspace(255, 0, 256)[:, None]
    assert "colour" in check_mri_input(Image.fromarray(a))


def test_a_blank_image_is_refused():
    assert "blank" in check_mri_input(Image.new("RGB", (256, 256), (40, 40, 40)))


def test_rgba_and_palette_images_load():
    base = Image.open(SAMPLES[0])
    for mode in ("RGBA", "P", "L"):
        buf = io.BytesIO()
        base.convert(mode).save(buf, format="PNG")
        buf.seek(0)
        assert load_mri_rgb(buf).mode == "RGB"
