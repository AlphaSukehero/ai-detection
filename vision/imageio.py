"""Load an MRI upload faithfully, and refuse what is not an MRI.

Two defects this module exists to prevent, both verified on 2026-09-30:

  * Image.convert("RGB") clips 16-bit pixel values to 255. A 16-bit export of
    a glioma became an almost white square and was classified "No Tumor" at
    100%. High-bit-depth images are windowed to 8 bits by percentile instead,
    exactly as DICOM already was.
  * The classifier answers every input with one of its four classes. Random
    noise was reported "No Tumor 100%". An input gate refuses images that
    cannot be a greyscale brain scan before the model sees them.

Gate thresholds were set against 300 scans from the training manifest
(colour spread <= 4.0, neighbour correlation >= 0.61, smallest side 150 px)
with a wide margin on each.
"""
import os

import numpy as np
from PIL import Image

DICOM_EXTENSIONS = {".dcm", ".dicom"}

MIN_SIDE_PX = 64
MAX_COLOUR_SPREAD = 25.0     # mean |R-G| + |G-B|; real scans <= 4
MIN_NEIGHBOUR_CORR = 0.4     # horizontal pixel correlation; noise ~ 0
MIN_CONTRAST_STD = 5.0       # grey-level std; a blank frame ~ 0


def window_to_uint8(arr, lo_pct=1.0, hi_pct=99.0):
    """Percentile-window a high-bit-depth array to uint8."""
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_rgb(arr8):
    if arr8.ndim == 2:
        return Image.fromarray(np.stack([arr8] * 3, axis=-1), "RGB")
    if arr8.ndim == 3 and arr8.shape[-1] == 1:
        return Image.fromarray(np.repeat(arr8, 3, axis=-1), "RGB")
    if arr8.ndim == 3 and arr8.shape[-1] >= 3:
        return Image.fromarray(np.ascontiguousarray(arr8[:, :, :3]), "RGB")
    raise ValueError("Unsupported pixel array shape.")


def load_dicom_rgb(path):
    """Read a DICOM file and return a windowed RGB PIL image."""
    try:
        import pydicom
    except ImportError as e:
        raise ValueError("DICOM support requires the 'pydicom' package.") from e
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return _to_rgb(window_to_uint8(arr * slope + intercept))


def load_mri_rgb(path_or_file):
    """Any supported image (8/16-bit, float, palette, RGBA, DICOM) -> RGB."""
    ext = ""
    if isinstance(path_or_file, (str, os.PathLike)):
        ext = os.path.splitext(str(path_or_file))[1].lower()
    if ext in DICOM_EXTENSIONS:
        return load_dicom_rgb(path_or_file)
    try:
        img = Image.open(path_or_file)
        img.load()
    except Exception as e:
        raise ValueError(
            f"Unsupported or corrupt image format '{ext or 'unknown'}': {e}") from e
    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        return _to_rgb(window_to_uint8(np.asarray(img)))
    return img.convert("RGB")


def check_mri_input(img):
    """None if the image can be a greyscale brain scan, else the reason."""
    w, h = img.size
    if min(w, h) < MIN_SIDE_PX:
        return (f"Image is too small ({w}×{h} px); at least "
                f"{MIN_SIDE_PX}×{MIN_SIDE_PX} px is needed for a reliable reading.")
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    g = a.mean(axis=2)
    if g.std() < MIN_CONTRAST_STD:
        return "The image is blank or has almost no contrast."
    corr = float(np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1])
    if corr < MIN_NEIGHBOUR_CORR:
        return ("The image looks like noise rather than anatomy, so it was "
                "not classified.")
    spread = float((np.abs(a[..., 0] - a[..., 1])
                    + np.abs(a[..., 1] - a[..., 2])).mean())
    if spread > MAX_COLOUR_SPREAD:
        return ("This is a colour image. MRI slices are greyscale; upload the "
                "scan itself, not a photo or a colour-mapped rendering.")
    return None
