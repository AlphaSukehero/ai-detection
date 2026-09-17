from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file,
    abort
)
import os
import re
import json
import time
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image


# Optional TensorFlow import with fallback
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import warning:", e)
    TF_AVAILABLE = False

from mlkit.registry import RegistryError, load_card, validate_card
from ecg.parameters import analyse as analyse_ecg_parameters, fs_from_scale
from ecg.digitize import extract_leads
from ecg.clinical import clinical_report
from ecg.beats import segment_signal
from reporting.pdf import build_pdf_report
from webapp.metadata import (
    NOT_PROVIDED, GENDER_OPTIONS, PATIENT_FIELDS, SURVEY_FIELDS,
    collect_metadata, finalize_metadata, metadata_rows, _clean_text,
)
from vision.gradcam import (
    generate_gradcam_heatmap, create_gradcam_overlay,
    create_tumor_region_highlight, calculate_tumor_area, calculate_tumor_size,
    calculate_tumor_location, calculate_severity, calculate_spread,
)

# Optional DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except Exception as e:
    print("pydicom import warning:", e)
    DICOM_AVAILABLE = False

# Optional OpenCV support (used only for ECG image digitization)
try:
    import cv2
    CV2_AVAILABLE = True
except Exception as e:
    print("OpenCV import warning:", e)
    CV2_AVAILABLE = False


# ============================================================
# FLASK APPLICATION SETUP
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs("static/samples", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit


# ============================================================
# MODEL PATHS & GLOBALS
# ============================================================

MODEL_FOLDER = "model"

# Every model below is loaded through mlkit.registry, which refuses any
# checkpoint whose sidecar card does not match the task, class order, input
# shape and preprocessing the caller expects. Two defects that shipped here
# before the registry existed -- a vgg16 model fed /255 inputs, and a
# brain-tumor model serving land cover -- become refused loads rather than
# confident nonsense.
MRI_MODEL_CANDIDATES = [
    (os.path.join(MODEL_FOLDER, "mri_vgg16.keras"), [224, 224, 3], "vgg16_preprocess_input"),
    (os.path.join(MODEL_FOLDER, "mri_cnn.keras"), [128, 128, 3], "rescale_255"),
]
MRI_CARD_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

SATELLITE_MODEL_FILE = os.path.join(MODEL_FOLDER, "satellite_best.keras")
SATELLITE_CARD_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
SATELLITE_INPUT_SHAPE = [64, 64, 3]

ECG_MODEL_FILE = os.path.join(MODEL_FOLDER, "ecg_cnn.keras")
ECG_CARD_CLASSES = ["N", "S", "V", "F", "Q"]
ECG_INPUT_SHAPE = [280, 1]


_brain_tumor_model = None
_satellite_model = None
_ecg_model = None


def _load_validated(path, task, classes, input_shape, preprocessing):
    """Load a model only if its card matches what the caller expects.

    Returns (model, card) on success and (None, None) otherwise, printing the
    reason. A mismatch is never silently tolerated: a model whose card does
    not match is the exact failure the registry exists to catch.
    """
    if not TF_AVAILABLE or not os.path.exists(path):
        return None, None
    try:
        card = load_card(path)
        validate_card(card, task, classes, input_shape, preprocessing)
        model = load_model(path, compile=False)
    except RegistryError as e:
        print(f"Refusing to load {path}: {e}")
        return None, None
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None
    units = model.output_shape[-1]
    if units != len(classes):
        print(f"Refusing to load {path}: outputs {units} classes, card declares {len(classes)}.")
        return None, None
    print(f"Loaded {path} ({task}, {units} classes, {preprocessing})")
    return model, card


def preprocess_mri(img_array, preprocessing="vgg16_preprocess_input"):
    """Prepare an RGB array for the brain tumor model it was trained for.

    The transform is read from the model card, never assumed: the vgg16
    transfer model needs ImageNet BGR mean subtraction, while the scratch CNN
    is trained on /255 inputs. Feeding one the other collapses every
    prediction to a single class.
    """
    batch = np.expand_dims(np.asarray(img_array, dtype=np.float32), axis=0)
    if preprocessing == "vgg16_preprocess_input" and TF_AVAILABLE:
        return vgg16_preprocess(batch.copy())
    return batch / 255.0


def _card_macro_f1(path):
    """Macro-F1 recorded on a candidate's card, or -1 if it has none.

    Ranking key for model selection. -1 rather than 0 so an unmeasured model
    always loses to a measured one, however badly the measured one scores:
    an unknown score is not evidence of a good one.
    """
    try:
        return float(load_card(path).get("metrics", {}).get("macro_f1", -1.0))
    except (RegistryError, ValueError, TypeError):
        return -1.0


def get_brain_tumor_model():
    """Return (model, card) for the best-measured validated MRI model.

    Candidates are ranked by the macro-F1 on their own cards, not by their
    position in MRI_MODEL_CANDIDATES. Hardcoded order is a standing hazard:
    it silently serves whichever checkpoint someone happened to list first,
    and the list gives no signal when a retrain makes the other one better.
    Macro-F1 rather than accuracy because these classes are imbalanced.
    """
    global _brain_tumor_model
    if _brain_tumor_model is None:
        _brain_tumor_model = (None, None)
        ranked = sorted(MRI_MODEL_CANDIDATES,
                        key=lambda c: _card_macro_f1(c[0]), reverse=True)
        for path, shape, prep in ranked:
            model, card = _load_validated(path, "mri", MRI_CARD_CLASSES, shape, prep)
            if model is not None:
                print(f"Selected {path} (macro_f1={_card_macro_f1(path):.4f})")
                _brain_tumor_model = (model, card)
                break
    return _brain_tumor_model


def get_satellite_model():
    """Return (model, card) for the EuroSAT land cover model, if installed.

    Returns (None, None) when no validated model exists, in which case
    predict_satellite() falls back to spectral analysis.
    """
    global _satellite_model
    if _satellite_model is None:
        _satellite_model = _load_validated(
            SATELLITE_MODEL_FILE, "satellite", SATELLITE_CARD_CLASSES,
            SATELLITE_INPUT_SHAPE, "rescale_255")
    return _satellite_model


def get_ecg_model():
    """Return (model, card) for the AAMI 5-class beat classifier, if installed."""
    global _ecg_model
    if _ecg_model is None:
        _ecg_model = _load_validated(
            ECG_MODEL_FILE, "ecg", ECG_CARD_CLASSES, ECG_INPUT_SHAPE, "beat_zscore")
    return _ecg_model


# ============================================================
# ECG HELPER FUNCTIONS
# ============================================================

def load_ecg_file(filepath):
    """Read an ECG data file into a numeric array.

    Delimiters are tried explicitly rather than sniffed: pandas' sep=None
    sniffer misreads a single-column decimal file (it treats "." as the
    separator), which breaks plain .txt / .dat exports.
    """
    extension = os.path.splitext(filepath)[1].lower()
    separators = [",", r"\s+", ";", "\t"]
    if extension != ".csv":
        separators = [r"\s+", ",", ";", "\t"]

    last_error = None
    for sep in separators:
        try:
            data = pd.read_csv(filepath, header=None, sep=sep, engine="python",
                               comment="#", skip_blank_lines=True)
        except Exception as e:
            last_error = e
            continue

        numeric = data.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if numeric.empty:
            continue

        # A header row (or other text) leaves an all-NaN first row; drop rows
        # that are still partially unparsed rather than feeding NaNs forward.
        numeric = numeric.dropna(axis=0, how="any")
        if numeric.empty:
            continue

        return numeric.values.astype(float)

    raise ValueError(
        "Unable to read ECG file: no numeric samples found. Expected a plain "
        f"CSV/TXT file of numeric ECG samples. ({last_error})"
    )


def prepare_beats(ecg_values, fs=360.0):
    """Segment a trace into R-peak-centred beats with their RR context.

    Previously this chopped the signal into consecutive 280-sample blocks,
    which bore no relation to how the model was trained: training windows are
    centred on the R peak at index 100. Segmentation now comes from
    ecg.beats, the same code scripts/prepare_ecg.py uses, so the two cannot
    drift apart -- and detecting the peaks is what makes the RR features
    available at inference at all.
    """
    signal = np.asarray(ecg_values, dtype=np.float32)
    if signal.ndim == 2:
        # A digitized multi-lead image arrives as (leads, samples); the model
        # is trained on a single lead.
        signal = signal[0] if signal.shape[0] < signal.shape[1] else signal[:, 0]
    beats, rr = segment_signal(signal.ravel(), fs)
    if len(beats) == 0:
        raise ValueError(
            "No heartbeats could be detected in this recording, so no beat "
            "classification is possible."
        )
    return beats, rr


ECG_BEAT_NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unclassifiable beat",
}


# A class whose measured F1 falls below this is not named in any output. The
# threshold is a judgement call, not a standard: it is set where a label stops
# being better than a coin-flip guess weighted by prevalence. It lives here,
# next to the code that applies it, so changing it is a visible decision.
MIN_REPORTABLE_F1 = 0.30


def _reliable_ecg_classes(card):
    """Classes this model's own card shows it can discriminate.

    Falls back to every class when a card records no per-class F1, because an
    older card means unmeasured, and suppressing everything would be a worse
    failure than the status quo -- but it is logged so it is not silent.
    """
    per_class = (card or {}).get("metrics", {}).get("per_class_f1")
    if not per_class:
        print("ECG card records no per-class F1; reporting all classes unfiltered.")
        return set(card.get("classes", []))
    return {c for c, f1 in per_class.items() if float(f1) >= MIN_REPORTABLE_F1}


class NoECGModelError(RuntimeError):
    """No validated ECG classifier is installed."""


def predict_ecg_cnn(X):
    """Classify each 280-sample beat with the AAMI 5-class 1D CNN.

    Returns None when no validated model is installed. There is no fallback
    classifier: an ECG with no model behind it gets no prediction at all.
    """
    model, card = get_ecg_model()
    if model is None:
        return None

    beats, rr = X
    beats = np.asarray(beats, dtype=np.float32).reshape(-1, 280)
    # beat_zscore: each beat is standardised on its own, exactly as in
    # training. ecg.beats already does this; repeating it is idempotent and
    # keeps the guarantee local to the code that feeds the model.
    mean = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    beats = (beats - mean) / std

    # A card declaring aux_inputs was trained with RR context; one without it
    # is an older morphology-only checkpoint and is still served correctly.
    # The card decides, never the calling code -- that is the whole point of
    # the registry.
    if card.get("aux_inputs", {}).get("rr"):
        width = int(card["aux_inputs"]["rr"])
        rr = np.asarray(rr, dtype=np.float32).reshape(-1, width)
        if len(rr) != len(beats):
            raise NoECGModelError(
                "Beat and RR-context arrays disagree in length; refusing to "
                "classify rather than pair a beat with another beat's rhythm."
            )
        preds = model.predict({"beat": beats[..., np.newaxis], "rr": rr}, verbose=0)
    else:
        preds = model.predict(beats[..., np.newaxis], verbose=0)
    classes = card["classes"]
    per_beat = preds.argmax(axis=1)
    mean_probs = preds.mean(axis=0)

    counts = {c: int((per_beat == i).sum()) for i, c in enumerate(classes)}
    total = len(per_beat) or 1
    abnormal = total - counts.get("N", 0)
    abnormal_fraction = abnormal / total

    # Which class names this model has earned the right to say. A class whose
    # measured F1 is near zero carries no information, so naming it as the
    # finding would be an invented specificity: the current card records
    # F=0.008 and Q=0.003, meaning those labels are essentially never right.
    # Such beats still count as "not normal" -- that part the model can do --
    # they just are not given a name.
    reliable = _reliable_ecg_classes(card)

    top_idx = int(np.argmax(mean_probs))
    top_class = classes[top_idx]
    if top_class == "N" and abnormal_fraction < 0.10:
        prediction, abnormal_type = "NORMAL", "No ectopic beats detected"
        confidence = float(mean_probs[classes.index("N")]) * 100.0
    else:
        ectopic = [c for c in classes if c != "N"]
        # A reliable class with no beats assigned to it is not a finding: the
        # ectopy is real but belongs to a class this model cannot name.
        nameable = [c for c in ectopic if c in reliable and counts[c] > 0]
        prediction = "ABNORMAL"
        if nameable:
            worst = max(nameable, key=lambda c: counts[c])
            abnormal_type = ECG_BEAT_NAMES[worst]
            confidence = max(abnormal_fraction,
                             float(mean_probs[classes.index(worst)])) * 100.0
        else:
            worst = max(ectopic, key=lambda c: counts[c])
            abnormal_type = ("Non-normal beats detected; this model cannot "
                             "reliably identify which type")
            confidence = abnormal_fraction * 100.0

    # The per-class distribution is shown only for classes the model can
    # actually discriminate; the rest are reported as one unnamed group so the
    # page never prints a precise-looking percentage against a class with an
    # F1 of 0.003.
    distribution = {ECG_BEAT_NAMES[c]: f"{float(mean_probs[i]) * 100:.2f}"
                    for i, c in enumerate(classes) if c in reliable}
    unreliable_mass = sum(float(mean_probs[i]) for i, c in enumerate(classes)
                          if c not in reliable)
    if unreliable_mass > 0:
        distribution["Other / not reliably classified"] = f"{unreliable_mass * 100:.2f}"

    s_probability = (f"{float(mean_probs[classes.index('S')]) * 100:.2f}"
                     if "S" in reliable else None)

    return {
        "prediction": prediction,
        "abnormal_type": abnormal_type,
        "confidence": f"{confidence:.2f}",
        "atrial_probability": s_probability,
        "beat_predictions": preds.max(axis=1),
        "beat_counts": counts,
        "beat_distribution": distribution,
        "beats_analyzed": total,
        "method": "AAMI 5-class 1D CNN (MIT-BIH, inter-patient split)",
    }


def predict_ecg(X):
    """Classify a strip, or refuse.

    Previously this fell back to an 8-qubit QCNN whose preprocessing was an
    unpickled PCA + MinMaxScaler fitted under a different scikit-learn minor
    version, and below that to sigmoid(std(X)) -- a number with no diagnostic
    meaning presented as a confidence. Both produced clinical-sounding output
    from nothing. A missing model is now an error, not a guess.
    """
    result = predict_ecg_cnn(X)
    if result is None:
        raise NoECGModelError(
            "No validated ECG classifier is installed, so no beat "
            "classification can be reported. Rhythm and interval measurements "
            "below are computed from the signal and remain valid."
        )
    return result


def create_ecg_plot(ecg_values, filename):
    signal = np.asarray(ecg_values, dtype=float).flatten()
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=1.2, color='#4f46e5')
    plt.title("ECG Signal Waveform Analysis", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Sample Index", fontsize=10)
    plt.ylabel("Normalized Amplitude", fontsize=10)
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()

    plot_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


# ============================================================
# IMAGE FORMAT HELPERS
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
                    ".tif", ".tiff", ".dcm", ".dicom"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
ECG_DATA_EXTENSIONS = {".csv", ".txt", ".dat", ".hea", ".xml", ".json"}


def load_dicom_as_pil(path):
    """Read a DICOM file and return a windowed RGB PIL image."""
    if not DICOM_AVAILABLE:
        raise ValueError("DICOM support requires the 'pydicom' package.")
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array, dtype=np.float32)

    # Apply modality rescaling (slope / intercept)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # Window to a sensible display range (1st - 99th percentile)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[:, :, :3]
    else:
        raise ValueError("Unsupported DICOM pixel array shape.")

    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def load_image_rgb(image_path):
    """Load any supported image format (PIL formats + DICOM) as an RGB PIL Image."""
    ext = os.path.splitext(str(image_path))[1].lower()
    if ext in DICOM_EXTENSIONS:
        return load_dicom_as_pil(image_path)
    try:
        img = Image.open(image_path)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Unsupported or corrupt image format '{ext or 'unknown'}': {e}") from e


# ============================================================
# BRAIN TUMOR MRI PREDICTION LOGIC
# ============================================================

MRI_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}


class NoMRIModelError(RuntimeError):
    """No validated brain-tumor classifier could produce a prediction."""


def predict_brain_tumor(image_path):
    img = load_image_rgb(image_path)
    model, card = get_brain_tumor_model()
    # Resize to whatever the card declares -- 224 for vgg16, 128 for the
    # scratch CNN. The Grad-CAM path below reuses this same array.
    size = tuple(card["input_shape"][:2]) if card else (224, 224)
    img_array = np.array(img.resize(size), dtype=np.float32)

    if model is None:
        raise NoMRIModelError(
            "No validated brain-tumor model is installed, so no classification "
            "can be reported."
        )
    try:
        input_tensor = preprocess_mri(img_array, card["preprocessing"])
        preds = model.predict(input_tensor)[0]
    except Exception as e:
        raise NoMRIModelError(
            f"The brain-tumor model failed to produce a prediction ({e})."
        ) from e
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100.0
    probs = {MRI_CLASSES[i]: float(preds[i]) * 100.0 for i in range(len(preds))}

    top_class = MRI_CLASSES[top_idx]

    # Clinical findings description based on predicted tumor type
    descriptions = {
        "Glioma": {
            "title": "Glioma Detected",
            "findings": "Hyperintense glial tissue lesion observed with irregular focal boundaries.",
            "recommendation": "Urgent neurosurgical consultation and contrast-enhanced MRI scan recommended."
        },
        "Meningioma": {
            "title": "Meningioma Detected",
            "findings": "Dural-based extra-axial lesion showing characteristic homogeneous enhancement.",
            "recommendation": "Neurosurgical assessment recommended to evaluate mass effect."
        },
        "No Tumor": {
            "title": "No Tumor Detected",
            "findings": "Brain parenchyma appears within normal limits. No obvious focal mass or structural lesion.",
            "recommendation": "Routine clinical follow-up as indicated by symptoms."
        },
        "Pituitary": {
            "title": "Pituitary Tumor Detected",
            "findings": "Sellar / suprasellar mass consistent with pituitary adenoma characteristics.",
            "recommendation": "Endocrine evaluation and dedicated thin-slice sellar MRI recommended."
        }
    }

    info = descriptions.get(top_class, descriptions["No Tumor"])

    return {
        "prediction": top_class,
        "confidence": f"{confidence:.2f}",
        "probabilities": {k: f"{v:.2f}" for k, v in probs.items()},
        "raw_probabilities": probs,
        "findings": info["findings"],
        "recommendation": info["recommendation"]
    }


# ============================================================
# SATELLITE IMAGE ANALYSIS LOGIC
# ============================================================

# The four display groups the UI and PDF reports are built around.
SATELLITE_CLASSES = {
    0: "Forest / Vegetation",
    1: "Urban / Built-up",
    2: "Water Body",
    3: "Agricultural / Barren Land"
}

# EuroSAT's 10 land cover classes collapsed onto those groups. The card's
# class order is authoritative; this maps by name, never by index.
SATELLITE_GROUPS = {
    "AnnualCrop": "Agricultural / Barren Land",
    "Forest": "Forest / Vegetation",
    "HerbaceousVegetation": "Forest / Vegetation",
    "Highway": "Urban / Built-up",
    "Industrial": "Urban / Built-up",
    "Pasture": "Agricultural / Barren Land",
    "PermanentCrop": "Agricultural / Barren Land",
    "Residential": "Urban / Built-up",
    "River": "Water Body",
    "SeaLake": "Water Body",
}
SATELLITE_GROUP_INDEX = {name: i for i, name in SATELLITE_CLASSES.items()}


def predict_satellite(image_path):
    img = load_image_rgb(image_path)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # Calculate Pseudo-NDVI: (Green - Red) / (Green + Red + 1e-5)
    ndvi_map = (g - r) / (g + r + 1e-5)
    mean_ndvi = float(np.mean(ndvi_map))

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    model, card = get_satellite_model()
    fine_probs = None
    if model is not None:
        try:
            size = tuple(card["input_shape"][:2])
            scene = np.array(img.resize(size), dtype=np.float32)
            preds = model.predict(np.expand_dims(scene / 255.0, axis=0))[0]
            classes = card["classes"]
            fine_probs = {classes[i]: float(preds[i]) * 100.0 for i in range(len(preds))}
            top_idx, confidence, probs = group_eurosat_probs(fine_probs)
        except Exception as e:
            print("Satellite model error, using spectral analysis fallback:", e)
            top_idx, confidence, probs = spectral_satellite_analysis(r_mean, g_mean, b_mean, mean_ndvi)
            fine_probs = None
    else:
        top_idx, confidence, probs = spectral_satellite_analysis(r_mean, g_mean, b_mean, mean_ndvi)

    top_class = SATELLITE_CLASSES[top_idx]

    # Environmental health & Land breakdown metrics
    if top_class == "Forest / Vegetation":
        veg_pct = round(65.0 + mean_ndvi * 35.0, 1)
        urban_pct = round(10.0, 1)
        water_pct = round(15.0, 1)
        barren_pct = round(100.0 - (veg_pct + urban_pct + water_pct), 1)
        env_score = "Good (88/100)"
    elif top_class == "Urban / Built-up":
        urban_pct = round(70.0 + (r_mean / 255.0) * 20.0, 1)
        veg_pct = round(15.0, 1)
        water_pct = round(5.0, 1)
        barren_pct = round(100.0 - (urban_pct + veg_pct + water_pct), 1)
        env_score = "Moderate (62/100)"
    elif top_class == "Water Body":
        water_pct = round(80.0 + (b_mean / 255.0) * 15.0, 1)
        veg_pct = round(10.0, 1)
        urban_pct = round(5.0, 1)
        barren_pct = round(100.0 - (water_pct + veg_pct + urban_pct), 1)
        env_score = "High Water Quality (92/100)"
    else:
        barren_pct = round(60.0, 1)
        veg_pct = round(25.0, 1)
        urban_pct = round(10.0, 1)
        water_pct = round(5.0, 1)
        env_score = "Fair (71/100)"

    # The dominant-class share is derived from the image, so the remainder can
    # overshoot; clamp to zero and renormalise so the four shares sum to 100%.
    shares = {
        "Vegetation": max(0.0, veg_pct),
        "Urban": max(0.0, urban_pct),
        "Water": max(0.0, water_pct),
        "Barren": max(0.0, barren_pct),
    }
    total = sum(shares.values()) or 1.0
    shares = {k: v / total * 100.0 for k, v in shares.items()}

    return {
        "prediction": top_class,
        "confidence": f"{confidence:.2f}",
        "ndvi": f"{mean_ndvi:.3f}",
        "environmental_health": env_score,
        "probabilities": {k: f"{v:.2f}" for k, v in probs.items()},
        "land_breakdown": {k: f"{v:.1f}%" for k, v in shares.items()},
        "eurosat_probabilities": ({k: f"{v:.2f}" for k, v in fine_probs.items()}
                                  if fine_probs else None),
        "method": ("EuroSAT 10-class CNN land cover classifier" if fine_probs
                   else "Spectral RGB / pseudo-NDVI analysis (no trained satellite model installed)"),
        "findings": f"Primary terrain classified as {top_class} with spectral NDVI index of {mean_ndvi:.3f}.",
        "recommendation": "Monitored for seasonal vegetation change and urban encroachment."
    }


def group_eurosat_probs(fine_probs):
    """Collapse EuroSAT's 10 class probabilities onto the 4 display groups.

    The winning group is the one with the highest summed probability, which is
    more stable than taking the group of the single top EuroSAT class.
    """
    grouped = {name: 0.0 for name in SATELLITE_CLASSES.values()}
    for cls, pct in fine_probs.items():
        grouped[SATELLITE_GROUPS[cls]] += pct
    top_class = max(grouped, key=grouped.get)
    return SATELLITE_GROUP_INDEX[top_class], grouped[top_class], grouped


def spectral_satellite_analysis(r_mean, g_mean, b_mean, mean_ndvi):
    if g_mean > r_mean and g_mean > b_mean and mean_ndvi > 0.05:
        top_idx = 0  # Forest
        probs = [82.5, 8.1, 4.2, 5.2]
    elif b_mean > r_mean and b_mean > g_mean * 0.9:
        top_idx = 2  # Water
        probs = [5.1, 6.3, 85.2, 3.4]
    elif r_mean > 120 and g_mean > 120 and b_mean > 120:
        top_idx = 1  # Urban
        probs = [10.2, 79.4, 3.1, 7.3]
    else:
        top_idx = 3  # Agriculture / Barren
        probs = [15.1, 12.3, 6.2, 66.4]

    confidence = probs[top_idx]
    probs_dict = {SATELLITE_CLASSES[i]: probs[i] for i in range(4)}
    return top_idx, confidence, probs_dict


# ============================================================
# ROUTE HANDLERS
# ============================================================

# ------------------------------------------------------------
# UPLOAD DIRECTORY POLICY
# ------------------------------------------------------------

# Every file this app writes into uploads/ is named by one of the templates
# below: a fixed prefix, a uuid4 hex stem, and a known extension. Serving is
# restricted to that shape so /uploads/<filename> can only ever return a file
# this app generated -- not, say, a .py or .keras that shares the directory
# after an operator mistake. send_from_directory already blocks traversal;
# this is about what is legitimately in the folder, not what is above it.
_UPLOAD_PREFIXES = ("ecg_upload", "ecg_waveform_", "mri_upload_", "mri_analysis_",
                    "mri_heatmap_", "mri_highlight_", "sat_upload_", "sat_analysis_")
# Derived from the extension sets the upload handlers actually accept, so a
# new accepted format cannot become an unservable file through a missed edit
# in a second hand-maintained list.
_UPLOAD_EXTS = sorted(e.lstrip(".") for e in
                      IMAGE_EXTENSIONS | ECG_DATA_EXTENSIONS | {"png"})
UPLOAD_NAME_RE = re.compile(
    "(?:%s)[0-9a-f]{8}\\.(?:%s)" % ("|".join(map(re.escape, _UPLOAD_PREFIXES)),
                                    "|".join(map(re.escape, _UPLOAD_EXTS)))
)
SAMPLE_NAME_RE = re.compile(r"sample_[a-z0-9_]{1,40}_ecg\.csv")

# Uploads are per-request scratch: a waveform plot is rendered, shown once,
# and never referenced again. Without a sweep the directory grows without
# bound and keeps patient-derived images on disk indefinitely.
UPLOAD_TTL_SECONDS = int(os.environ.get("UPLOAD_TTL_SECONDS", 24 * 3600))


def _is_generated_name(filename):
    return bool(UPLOAD_NAME_RE.fullmatch(filename))


def sweep_uploads(ttl_seconds=None, now=None):
    """Delete generated upload files older than the TTL. Returns the count.

    Only files matching UPLOAD_NAME_RE are eligible, so anything an operator
    deliberately placed in the folder is left alone.
    """
    ttl = UPLOAD_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    now = time.time() if now is None else now
    folder = app.config["UPLOAD_FOLDER"]
    removed = 0
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    for name in names:
        if not _is_generated_name(name):
            continue
        path = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(path) > ttl:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


@app.before_request
def _sweep_uploads_periodically():
    """Sweep at most once every 10 minutes, on whatever request comes first.

    A background thread would be tidier but this app is deliberately a single
    synchronous process; hanging the sweep off request traffic keeps it that
    way and costs a directory listing per ten minutes.
    """
    global _last_sweep
    now = time.time()
    if now - _last_sweep < 600:
        return
    _last_sweep = now
    sweep_uploads(now=now)


_last_sweep = 0.0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ecg")
def ecg():
    return render_template("ecg.html", patient={}, patient_fields=PATIENT_FIELDS,
                           gender_options=GENDER_OPTIONS)


@app.route("/brain-tumor")
def brain_tumor():
    return render_template("brain_tumor.html", patient={}, patient_fields=PATIENT_FIELDS,
                           gender_options=GENDER_OPTIONS)


@app.route("/satellite")
def satellite():
    return render_template("satellite.html", survey={}, survey_fields=SURVEY_FIELDS)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    if not _is_generated_name(filename):
        abort(404)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/samples/<filename>")
def sample_file(filename):
    if not SAMPLE_NAME_RE.fullmatch(filename):
        abort(404)
    return send_from_directory("static/samples", filename)


# ------------------------------------------------------------
# ANALYZE ECG
# ------------------------------------------------------------

ECG_QUALITY_PARAMS = ("heart_rate", "rhythm", "pr_interval", "qrs_duration",
                      "qt_interval", "qtc", "st_segment", "axis")


def _signal_quality(report):
    """Summarise how much of the ECG was actually measurable.

    Derived from the per-parameter quality flags across the 8 clinical
    parameters, not from the presence of a heart rate alone - a strip whose
    PR, QT, ST and axis are all unavailable is not "Good".

    Cut-offs: "Good" needs at least 6 of 8 parameters measured with none of
    them low-confidence downgrading the majority (>= 6 OK); "Partial" needs
    at least 3 measured at any confidence; below that, "Insufficient data".
    """
    from ecg.quality import OK, UNAVAILABLE
    flags = [report[k].quality for k in ECG_QUALITY_PARAMS if k in report]
    n_ok = sum(1 for f in flags if f == OK)
    n_measured = sum(1 for f in flags if f != UNAVAILABLE)
    if n_ok >= 6:
        return "Good"
    if n_measured >= 3:
        return "Partial"
    return "Insufficient data"


@app.route("/analyze_ecg", methods=["POST"])
def analyze_ecg():
    file = request.files.get("ecg_file")
    sample_type = request.form.get("sample_type")

    entered, meta_errors = collect_metadata(request.form, PATIENT_FIELDS)
    if meta_errors:
        return render_template("ecg.html", error=" ".join(meta_errors),
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)
    patient = finalize_metadata(entered, PATIENT_FIELDS)

    filepath = None
    file_ext = None
    if file and file.filename != "":
        raw_name = os.path.basename(file.filename)
        file_ext = os.path.splitext(raw_name)[1].lower()
        safe_name = "ecg_upload" + uuid.uuid4().hex[:8] + file_ext
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)
    elif sample_type:
        sample_path = os.path.join("static/samples", f"sample_{sample_type}_ecg.csv")
        if os.path.exists(sample_path):
            filepath = sample_path
            file_ext = ".csv"

    if filepath is None:
        return render_template("ecg.html",
                               error="Please select or upload an ECG data file or image.",
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)

    is_image = file_ext in IMAGE_EXTENSIONS

    try:
        if is_image:
            digitized = extract_leads(filepath)
            leads = digitized["leads"]
            ecg_values = leads.get("II", next(iter(leads.values())))
            # A digitized trace has one sample per pixel COLUMN, so its true
            # sampling rate is set by the paper speed: 25 mm/s x px/mm. It is
            # NOT the 360 Hz of the MIT-BIH CSV path. With no grid there is no
            # timebase at all; analyse() then reports everything unavailable.
            # Derived before segmentation, because R-peak detection uses fs
            # for its refractory window.
            px_per_mm = digitized["px_per_mm"]
            image_fs = fs_from_scale(px_per_mm) if px_per_mm is not None else 360.0
            X = prepare_beats(ecg_values, fs=image_fs)
            report = analyse_ecg_parameters(leads, fs=image_fs,
                                            px_per_mm=px_per_mm,
                                            from_image=True)
            input_source = f"ECG image ({digitized['layout'].replace('_', ' ')})"
        else:
            ecg_values = load_ecg_file(filepath)
            X = prepare_beats(ecg_values, fs=360.0)
            report = analyse_ecg_parameters(ecg_values, fs=360.0)
            input_source = "ECG data file"

        # Beat classification is optional: the signal measurements below stand
        # on their own, so a missing classifier degrades the page rather than
        # failing it.
        try:
            beat_result = predict_ecg(X)
            classifier_error = None
        except NoECGModelError as e:
            beat_result = None
            classifier_error = str(e)

        # Structured clinical reading: formula, value, reference range and
        # verdict per parameter. Built from the same Measurements shown above,
        # so the two can never disagree.
        clinical = clinical_report(report, sex=patient.get("gender"),
                                   beat_result=beat_result)

        plot_filename = "ecg_waveform_" + uuid.uuid4().hex[:8] + ".png"
        create_ecg_plot(ecg_values, plot_filename)

        if beat_result is None:
            classifier = None
            interpretation = ("No beat classification was produced: " + classifier_error)
            recommendation = ("Interval and rhythm measurements below are derived "
                              "from the signal itself and are unaffected. Any "
                              "diagnostic conclusion requires review by a clinician.")
        else:
            classifier = beat_result["method"]
            if beat_result["prediction"] == "NORMAL":
                interpretation = (f"The {classifier} classified the ECG as NORMAL "
                                  f"({beat_result['abnormal_type']}).")
                recommendation = "Normal rhythm detected. Routine clinical monitoring recommended."
            else:
                interpretation = (f"The {classifier} classified the ECG as ABNORMAL "
                                  f"({beat_result['abnormal_type']}).")
                recommendation = "Abnormal pattern identified. Clinical evaluation by a cardiologist is advised."

        result = {
            "prediction": beat_result["prediction"] if beat_result else None,
            "abnormal_type": beat_result["abnormal_type"] if beat_result else None,
            "confidence": beat_result["confidence"] if beat_result else None,
            "atrial_probability": beat_result["atrial_probability"] if beat_result else None,
            "classifier": classifier,
            "classifier_error": classifier_error,
            "beat_distribution": beat_result.get("beat_distribution") if beat_result else None,
            "beat_counts": beat_result.get("beat_counts") if beat_result else None,
            "beats_analyzed": beat_result.get("beats_analyzed") if beat_result else None,
            "heart_rate": report["display"]["heart_rate"],
            "rr_interval": report["display"]["rr_interval"],
            "qrs_duration": report["display"]["qrs_duration"],
            "pr_interval": report["display"]["pr_interval"],
            "qt_interval": report["display"]["qt_interval"],
            "qtc": report["display"]["qtc"],
            "sdnn": report["display"]["sdnn"],
            "rmssd": report["display"]["rmssd"],
            "rhythm": report["display"]["rhythm"],
            "st_segment": report["display"]["st_segment"],
            "axis": report["display"]["axis"],
            "signal_quality": _signal_quality(report),
            "interpretation": interpretation,
            "recommendation": recommendation,
            "clinical": clinical,
            "waveform": plot_filename,
            "input_source": input_source
        }

        return render_template("ecg.html", result=result, patient=patient,
                               patient_fields=PATIENT_FIELDS,
                               patient_rows=metadata_rows(patient, PATIENT_FIELDS),
                               gender_options=GENDER_OPTIONS)

    except Exception as e:
        print("ECG Analysis Error:", e)
        return render_template("ecg.html", error=f"ECG analysis failed: {str(e)}",
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)


# ------------------------------------------------------------
# ANALYZE BRAIN TUMOR
# ------------------------------------------------------------

@app.route("/analyze_brain_tumor", methods=["POST"])
def analyze_brain_tumor():
    file = request.files.get("mri_file")
    sample_name = request.form.get("sample_name")

    entered, meta_errors = collect_metadata(request.form, PATIENT_FIELDS)
    if meta_errors:
        return render_template("brain_tumor.html", error=" ".join(meta_errors),
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)
    patient = finalize_metadata(entered, PATIENT_FIELDS)

    filepath = None
    display_filename = None

    if file and file.filename != "":
        raw_name = os.path.basename(file.filename)
        ext = os.path.splitext(raw_name)[1].lower()
        safe_name = "mri_upload_" + uuid.uuid4().hex[:8] + ext
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)
    elif sample_name:
        sample_path = os.path.join("static/samples", f"mri_{sample_name}.jpg")
        if os.path.exists(sample_path):
            filepath = sample_path

    if filepath is None:
        return render_template("brain_tumor.html",
                               error="Please upload an MRI image or choose a sample.",
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)

    try:
        # Load image in any supported format (including DICOM)
        img = load_image_rgb(filepath)
        try:
            res = predict_brain_tumor(filepath)
        except NoMRIModelError as e:
            # Nothing further on this page is meaningful without a
            # classification -- the Grad-CAM, the tumour morphometry and the
            # clinical wording are all downstream of it -- so this is a
            # refusal, not a degraded render.
            return render_template("brain_tumor.html",
                                   error=str(e), patient=entered,
                                   patient_fields=PATIENT_FIELDS,
                                   gender_options=GENDER_OPTIONS)

        # Save a browser-renderable copy of the scan
        display_filename = "mri_analysis_" + uuid.uuid4().hex[:8] + ".png"
        img.save(os.path.join(app.config["UPLOAD_FOLDER"], display_filename))

        # Generate Grad-CAM heatmap if tumor detected
        heatmap_filename = None
        highlighted_filename = None
        area = 0.0
        width = 0
        height = 0
        location = "Not applicable"
        severity = "Not applicable"
        spread = "Not applicable"

        if res["prediction"] != "No Tumor":
            try:
                model, card = get_brain_tumor_model()
                if model is not None:
                    # Prepare image exactly as the card specifies
                    size = tuple(card["input_shape"][:2])
                    img_array = np.array(img.resize(size), dtype=np.float32)
                    input_tensor = preprocess_mri(img_array, card["preprocessing"])
                    
                    # Get predicted class index
                    predicted_class = list(MRI_CLASSES.values()).index(res["prediction"])
                    
                    # Generate Grad-CAM heatmap
                    heatmap = generate_gradcam_heatmap(model, input_tensor, predicted_class)
                    
                    if heatmap is not None:
                        # Create heatmap overlay image
                        gradcam_image = create_gradcam_overlay(img, heatmap)
                        heatmap_filename = "mri_heatmap_" + uuid.uuid4().hex[:8] + ".png"
                        gradcam_image.save(os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename))
                        
                        # Create highlighted tumor region
                        region_image = create_tumor_region_highlight(img, heatmap)
                        highlighted_filename = "mri_highlight_" + uuid.uuid4().hex[:8] + ".png"
                        region_image.save(os.path.join(app.config["UPLOAD_FOLDER"], highlighted_filename))
                        
                        # Calculate analysis metrics
                        threshold = np.percentile(heatmap, 90)
                        tumor_mask = (heatmap >= threshold).astype(np.uint8) * 255
                        
                        if CV2_AVAILABLE:
                            kernel = np.ones((5, 5), np.uint8)
                            tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
                            tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)
                            
                            contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                area = calculate_tumor_area(tumor_mask)
                                width, height = calculate_tumor_size(largest_contour)
                                location = calculate_tumor_location(largest_contour)
                                severity = calculate_severity(area)
                                spread = calculate_spread(area)
                        
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")

        result = {
            "image_filename": display_filename,
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "raw_probabilities": res["raw_probabilities"],
            "findings": res["findings"],
            "recommendation": res["recommendation"],
            "heatmap_filename": heatmap_filename,
            "highlighted_filename": highlighted_filename,
            "area": f"{area:.2f}",
            "width": width,
            "height": height,
            "location": location,
            "severity": severity,
            "spread": spread
        }

        return render_template("brain_tumor.html", result=result, patient=patient,
                               patient_fields=PATIENT_FIELDS,
                               patient_rows=metadata_rows(patient, PATIENT_FIELDS),
                               gender_options=GENDER_OPTIONS)

    except Exception as e:
        print("Brain tumor analysis error:", e)
        return render_template("brain_tumor.html", error=f"Analysis failed: {str(e)}",
                               patient=entered, patient_fields=PATIENT_FIELDS,
                               gender_options=GENDER_OPTIONS)


# ------------------------------------------------------------
# ANALYZE SATELLITE IMAGE
# ------------------------------------------------------------

@app.route("/analyze_satellite", methods=["POST"])
def analyze_satellite():
    file = request.files.get("satellite_file")
    sample_name = request.form.get("sample_name")

    entered, meta_errors = collect_metadata(request.form, SURVEY_FIELDS)
    if meta_errors:
        return render_template("satellite.html", error=" ".join(meta_errors),
                               survey=entered, survey_fields=SURVEY_FIELDS)
    survey = finalize_metadata(entered, SURVEY_FIELDS)

    filepath = None
    display_filename = None

    if file and file.filename != "":
        raw_name = os.path.basename(file.filename)
        ext = os.path.splitext(raw_name)[1].lower()
        safe_name = "sat_upload_" + uuid.uuid4().hex[:8] + ext
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)
    elif sample_name:
        # Satellite samples are stored separately from the MRI samples.
        for candidate in (f"satellite_{sample_name}.jpg", f"satellite_{sample_name}.png"):
            candidate_path = os.path.join("static/samples", candidate)
            if os.path.exists(candidate_path):
                filepath = candidate_path
                break
        if filepath is None:
            return render_template(
                "satellite.html",
                error=f"Sample image '{sample_name}' is not installed. Please upload your own image.",
                survey=entered, survey_fields=SURVEY_FIELDS)

    if filepath is None:
        return render_template("satellite.html", error="Please upload a satellite image.",
                               survey=entered, survey_fields=SURVEY_FIELDS)

    try:
        img = load_image_rgb(filepath)
        res = predict_satellite(filepath)

        # Save a browser-renderable copy of the processed image
        display_filename = "sat_analysis_" + uuid.uuid4().hex[:8] + ".png"
        img.save(os.path.join(app.config["UPLOAD_FOLDER"], display_filename))

        result = {
            "image_filename": display_filename,
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "ndvi": res["ndvi"],
            "environmental_health": res["environmental_health"],
            "probabilities": res["probabilities"],
            "land_breakdown": res["land_breakdown"],
            "findings": res["findings"],
            "recommendation": res["recommendation"],
            "method": res["method"]
        }

        return render_template("satellite.html", result=result, survey=survey,
                               survey_fields=SURVEY_FIELDS,
                               survey_rows=metadata_rows(survey, SURVEY_FIELDS))

    except Exception as e:
        print("Satellite analysis error:", e)
        return render_template("satellite.html",
                               error=f"Satellite image analysis failed: {str(e)}",
                               survey=entered, survey_fields=SURVEY_FIELDS)


# ------------------------------------------------------------
# PDF REPORT BUILDER -- document assembly lives in reporting/pdf.py
# ------------------------------------------------------------

def _form_meta(form, fields):
    """Rebuild a metadata dict from the hidden fields posted by a result page."""
    meta = {key: (_clean_text(form.get(key), 1000) or NOT_PROVIDED) for key, _ in fields}
    meta["report_id"] = _clean_text(form.get("report_id")) or ("RPT-" + uuid.uuid4().hex[:10].upper())
    meta["generated_at"] = _clean_text(form.get("generated_at")) or datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return meta


def _slug(value, fallback):
    """Filename-safe token derived from a metadata value."""
    if not value or value == NOT_PROVIDED:
        return fallback
    cleaned = "".join(c if c.isalnum() else "_" for c in value).strip("_")
    return cleaned[:40] or fallback


# ------------------------------------------------------------
# DOWNLOAD REPORTS (PDF)
# ------------------------------------------------------------

def _pdf_value(raw):
    """Render an unmeasured parameter explicitly in the PDF.

    A dash in a clinical report is ambiguous - it could mean zero, or missing.
    "Not measurable" says which.
    """
    if raw is None or raw.strip() in {"", "—", "-"}:
        return "Not measurable"
    return raw


def _clinical_from_form(form):
    """Rebuild the structured reading from the measurements posted back.

    Returns None when the page posted no measurement payload -- an older
    result page, or a direct post -- so the report simply omits the section
    rather than inventing one.
    """
    raw = form.get("clinical_json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "parameters" not in data:
        return None
    return data


def _ecg_classification_rows(form):
    """Rows for the classification table, or an explicit statement of absence.

    A report must never imply a classification happened when it did not, so
    when no classifier ran the table carries the reason instead of a row of
    em-dashes that reads like a missing measurement.
    """
    prediction = (form.get("prediction") or "").strip()
    if not prediction:
        return [
            ("Parameter", "Result"),
            ("Overall Rhythm Classification", "Not performed"),
            ("Reason", form.get("classifier_error")
                       or "No validated ECG classifier was installed."),
            ("Signal Quality", form.get("signal_quality", NOT_PROVIDED)),
            ("Input Source", form.get("input_source", NOT_PROVIDED)),
        ]
    return [
        ("Parameter", "Result"),
        ("Overall Rhythm Classification", prediction),
        ("Abnormality Class", form.get("abnormal_type", NOT_PROVIDED)),
        ("Classifier", form.get("classifier", NOT_PROVIDED)),
        ("Classifier Confidence", f"{form.get('confidence', '—')}%"),
        ("Supraventricular (S) Beat Probability", f"{form.get('atrial_probability', '—')}%"),
        ("Signal Quality", form.get("signal_quality", NOT_PROVIDED)),
        ("Input Source", form.get("input_source", NOT_PROVIDED)),
    ]


@app.route("/download_ecg_report", methods=["POST"])
def download_ecg_report():
    form = request.form
    meta = _form_meta(form, PATIENT_FIELDS)

    sections = [
        ("table", "1. Beat Classification Result", _ecg_classification_rows(form)),
        ("table", "2. ECG Waveform Parameters", [
            ("Measurement", "Value"),
            ("Heart Rate", _pdf_value(form.get("heart_rate"))),
            ("RR Interval", _pdf_value(form.get("rr_interval"))),
            ("QRS Duration", _pdf_value(form.get("qrs_duration"))),
            ("PR Interval", _pdf_value(form.get("pr_interval"))),
            ("QT Interval", _pdf_value(form.get("qt_interval"))),
            ("QTc (Corrected)", _pdf_value(form.get("qtc"))),
            ("HRV SDNN", _pdf_value(form.get("sdnn"))),
            ("HRV RMSSD", _pdf_value(form.get("rmssd"))),
            ("Rhythm", _pdf_value(form.get("rhythm"))),
            ("ST Segment", _pdf_value(form.get("st_segment"))),
            ("QRS Axis", _pdf_value(form.get("axis"))),
        ]),
        ("text", "3. Clinical Interpretation", form.get("interpretation", NOT_PROVIDED)),
        ("text", "4. Recommended Next Steps", form.get("recommendation", NOT_PROVIDED)),
    ]

    # The structured reading is carried from the result page as JSON. It is
    # display data echoed back, not a re-measurement: the report can only be
    # as trustworthy as the page that produced it. It is escaped like every
    # other field on the way into the PDF, and a malformed or absent payload
    # omits the section rather than substituting anything.
    clinical = _clinical_from_form(form)
    if clinical:
        sections.insert(1, ("table", "2. Parameters, Formulas & Reference Ranges",
                            [("Parameter", "Value / Range / Verdict")] +
                            [(r["label"],
                              f"{r['value']}  |  normal {r['range']}  |  "
                              f"{r['verdict'] or (r['reason'] or 'not measurable')}")
                             for r in clinical["parameters"]]))
        sections.append(("text", "5. Diagnostic Status",
                         f"{clinical['status']['classification']} — "
                         f"{clinical['status']['abnormality']}"))
        if clinical["status"]["findings"]:
            sections.append(("table", "6. Supporting Findings",
                             [("#", "Finding")] +
                             [(str(i + 1), f) for i, f in
                              enumerate(clinical["status"]["findings"])]))
        sections.append(("table", "7. Precautions & Next Steps",
                         [("#", "Action")] +
                         [(str(i + 1), p) for i, p in
                          enumerate(clinical["precautions"])]))

    buffer = build_pdf_report(
        title="ECG ANALYSIS REPORT",
        subtitle="AI Diagnostic Summary — AAMI 5-class beat classifier + signal measurements",
        accent="#4f46e5",
        meta=meta,
        meta_fields=PATIENT_FIELDS,
        meta_heading="Patient & Study Details",
        sections=sections,
        disclaimer=(
            "Disclaimer: This report is generated by a research prototype and is not a "
            "medical device. All findings, measurements and recommendations are "
            "model-derived estimates and must be verified by a qualified cardiologist "
            "before any clinical decision is made."
        ),
        footer_text="Unified AI Diagnostic Platform — ECG Analysis (research use only)"
    )

    name = _slug(meta.get("patient_id"), _slug(meta.get("patient_name"), "unidentified"))
    return send_file(buffer, as_attachment=True,
                     download_name=f"ECG_Report_{name}.pdf",
                     mimetype="application/pdf")


@app.route("/download_brain_tumor_report", methods=["POST"])
def download_brain_tumor_report():
    form = request.form
    meta = _form_meta(form, PATIENT_FIELDS)
    prediction = form.get("prediction", NOT_PROVIDED)

    sections = [
        ("table", "1. Classification Result", [
            ("Diagnostic Attribute", "AI System Evaluation"),
            ("Primary Tumor Classification", prediction),
            ("Model Confidence Score", f"{form.get('confidence', '—')}%"),
            ("Model", "VGG16 transfer-learning classifier (4-class)"),
        ]),
    ]

    if prediction != "No Tumor":
        area_val = form.get('area', '—')
        width_val = form.get('width', '—')
        height_val = form.get('height', '—')
        location_val = form.get("location", NOT_PROVIDED)
        severity_val = form.get("severity", NOT_PROVIDED)
        spread_val = form.get("spread", NOT_PROVIDED)
        sections.append(("table", "2. Lesion Analysis Metrics (Grad-CAM derived)", [
            ("Metric", "Estimated Value"),
            ("Area of Activation", f"{area_val}% of image"),
            ("Bounding Width", f"{width_val} px"),
            ("Bounding Height", f"{height_val} px"),
            ("Approximate Location", location_val),
            ("Severity Indicator", severity_val),
            ("Spread Indicator", spread_val),
        ]))

    sections.append(("text", "3. Radiological Findings Summary",
                     form.get("findings", NOT_PROVIDED)))
    sections.append(("text", "4. Recommended Next Steps",
                     form.get("recommendation", NOT_PROVIDED)))

    buffer = build_pdf_report(
        title="BRAIN TUMOR MRI DIAGNOSTIC REPORT",
        subtitle="AI Multi-Class MRI Classification with Grad-CAM Localization",
        accent="#7c3aed",
        meta=meta,
        meta_fields=PATIENT_FIELDS,
        meta_heading="Patient & Study Details",
        sections=sections,
        disclaimer=(
            "Disclaimer: This report is generated by a research prototype and is not a "
            "medical device. Lesion metrics are derived from model activation maps, not "
            "from calibrated radiological measurement, and must be validated by a "
            "board-certified radiologist."
        ),
        footer_text="Unified AI Diagnostic Platform — Brain Tumor MRI Analysis (research use only)"
    )

    name = _slug(meta.get("patient_id"), _slug(meta.get("patient_name"), "unidentified"))
    return send_file(buffer, as_attachment=True,
                     download_name=f"MRI_Report_{name}.pdf",
                     mimetype="application/pdf")


@app.route("/download_satellite_report", methods=["POST"])
def download_satellite_report():
    form = request.form
    meta = _form_meta(form, SURVEY_FIELDS)

    sections = [
        ("table", "1. Land Cover Classification", [
            ("Terrain Analysis Attribute", "Value / Assessment"),
            ("Dominant Land Cover Class", form.get("prediction", NOT_PROVIDED)),
            ("Classification Confidence", f"{form.get('confidence', '—')}%"),
            ("Vegetation Index (pseudo-NDVI)", form.get("ndvi", "—")),
            ("Environmental Health Status", form.get("health", NOT_PROVIDED)),
            ("Classification Method", form.get("method", NOT_PROVIDED)),
        ]),
        ("table", "2. Estimated Land Cover Breakdown", [
            ("Cover Type", "Share of Scene"),
            ("Vegetation", form.get("veg_pct", "—")),
            ("Urban / Built-up", form.get("urban_pct", "—")),
            ("Water", form.get("water_pct", "—")),
            ("Barren / Agricultural", form.get("barren_pct", "—")),
        ]),
        ("text", "3. Analysis Findings", form.get("findings", NOT_PROVIDED)),
        ("text", "4. Recommended Monitoring", form.get("recommendation", NOT_PROVIDED)),
    ]

    buffer = build_pdf_report(
        title="SATELLITE LAND COVER ANALYSIS REPORT",
        subtitle="Multi-Spectral Remote Sensing & Terrain Classification",
        accent="#2563eb",
        meta=meta,
        meta_fields=SURVEY_FIELDS,
        meta_heading="Survey & Acquisition Details",
        sections=sections,
        disclaimer=(
            "Disclaimer: Land cover shares and the vegetation index are estimated from "
            "RGB imagery using a pseudo-NDVI approximation, not from calibrated "
            "multi-spectral bands. Treat all values as indicative only."
        ),
        footer_text="Unified AI Diagnostic Platform — Satellite Land Cover Analysis"
    )

    name = _slug(meta.get("survey_id"), _slug(meta.get("site_name"), "survey"))
    return send_file(buffer, as_attachment=True,
                     download_name=f"Satellite_Report_{name}.pdf",
                     mimetype="application/pdf")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)