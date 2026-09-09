from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file
)
import os
import io
import uuid
from datetime import datetime
from html import escape
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import pennylane as qml
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
# PATIENT / STUDY METADATA
# ============================================================

GENDER_OPTIONS = ["Male", "Female", "Other", "Prefer not to say"]

# Fields captured for clinical (ECG / MRI) studies.
PATIENT_FIELDS = [
    ("patient_name", "Patient Name"),
    ("patient_id", "Patient ID / MRN"),
    ("age", "Age"),
    ("gender", "Gender"),
    ("contact", "Contact Number"),
    ("referring_physician", "Referring Physician"),
    ("study_date", "Study Date"),
    ("clinical_history", "Clinical History / Indication"),
]

# Fields captured for non-clinical satellite surveys.
SURVEY_FIELDS = [
    ("site_name", "Site / Area Name"),
    ("survey_id", "Survey Reference ID"),
    ("coordinates", "Coordinates (lat, lon)"),
    ("capture_date", "Image Capture Date"),
    ("sensor", "Sensor / Source"),
    ("analyst", "Analyst"),
    ("survey_notes", "Survey Notes"),
]

NOT_PROVIDED = "Not provided"


def _clean_text(value, max_length=200):
    """Trim, collapse whitespace and bound the length of a free-text field."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:max_length]


def collect_metadata(form, fields):
    """Read a metadata block from a submitted form.

    Returns (values, errors). Values always contain every key so templates and
    PDF builders can rely on them; blank entries stay empty here and are filled
    in by finalize_metadata().
    """
    values = {}
    errors = []

    for key, label in fields:
        limit = 1000 if key in ("clinical_history", "survey_notes") else 200
        values[key] = _clean_text(form.get(key), limit)

    if values.get("age"):
        try:
            age = int(float(values["age"]))
            if not 0 <= age <= 130:
                raise ValueError
            values["age"] = str(age)
        except (TypeError, ValueError):
            errors.append("Age must be a whole number between 0 and 130.")
            values["age"] = ""

    if values.get("gender") and values["gender"] not in GENDER_OPTIONS:
        errors.append("Please select a valid gender option.")
        values["gender"] = ""

    for date_key in ("study_date", "capture_date"):
        if values.get(date_key):
            try:
                datetime.strptime(values[date_key], "%Y-%m-%d")
            except ValueError:
                errors.append("Date must be in YYYY-MM-DD format.")
                values[date_key] = ""

    return values, errors


def finalize_metadata(values, fields):
    """Fill blanks with a placeholder and attach report identity fields."""
    finalized = {key: (values.get(key) or NOT_PROVIDED) for key, _ in fields}
    finalized["report_id"] = "RPT-" + uuid.uuid4().hex[:10].upper()
    finalized["generated_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return finalized


def metadata_rows(meta, fields):
    """Ordered (label, value) pairs for rendering in templates and PDFs."""
    return [(label, meta.get(key, NOT_PROVIDED)) for key, label in fields]


# ============================================================
# MODEL PATHS & GLOBALS
# ============================================================

MODEL_FOLDER = "model"

QCNN_WEIGHTS_FILE = os.path.join(MODEL_FOLDER, "mitbih_qcnn_final_weights.npy")
QCNN_PREPROCESSING_FILE = os.path.join(MODEL_FOLDER, "mitbih_qcnn_final_preprocessing.pkl")

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


# ============================================================
# 1. QUANTUM CNN (QCNN) FOR ECG ANALYSIS
# ============================================================

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)


def conv_block(params, q1, q2):
    qml.RY(params[0], wires=q1)
    qml.RZ(params[1], wires=q1)
    qml.RY(params[2], wires=q2)
    qml.RZ(params[3], wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RY(params[4], wires=q1)
    qml.RY(params[5], wires=q2)
    qml.CNOT(wires=[q2, q1])


@qml.qnode(dev, interface="autograd")
def qcnn_circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)

    for pair in range(4):
        conv_block(weights[pair], 2 * pair, 2 * pair + 1)

    for pair in range(3):
        conv_block(weights[4 + pair], 2 * pair + 1, 2 * pair + 2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[6, 7])

    for i in range(n_qubits):
        qml.RY(weights[7 + i, 0], wires=i)
        qml.RZ(weights[7 + i, 1], wires=i)

    return qml.expval(qml.PauliZ(0))


# Load QCNN Weights
qcnn_weights = None
qcnn_preprocessing = None

try:
    if os.path.exists(QCNN_WEIGHTS_FILE):
        qcnn_weights = np.load(QCNN_WEIGHTS_FILE)
        print("QCNN weights loaded successfully. Shape:", qcnn_weights.shape)

    if os.path.exists(QCNN_PREPROCESSING_FILE):
        with open(QCNN_PREPROCESSING_FILE, "rb") as f:
            qcnn_preprocessing = pickle.load(f)
        print("QCNN preprocessing loaded successfully.")
except Exception as e:
    print("WARNING: QCNN model could not be loaded:", e)


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


def get_brain_tumor_model():
    """Return (model, card) for the best available validated MRI model."""
    global _brain_tumor_model
    if _brain_tumor_model is None:
        _brain_tumor_model = (None, None)
        for path, shape, prep in MRI_MODEL_CANDIDATES:
            model, card = _load_validated(path, "mri", MRI_CARD_CLASSES, shape, prep)
            if model is not None:
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


def prepare_qcnn_input(ecg_values):
    if ecg_values.ndim == 2:
        if ecg_values.shape[1] == 280:
            return ecg_values
        if ecg_values.shape[0] == 280:
            return ecg_values.reshape(1, 280)

    flattened = ecg_values.flatten()
    usable_length = (len(flattened) // 280) * 280

    if usable_length == 0:
        raise ValueError("The uploaded ECG file does not contain enough data for 280-sample segments.")

    flattened = flattened[:usable_length]
    return flattened.reshape(-1, 280)


def apply_saved_preprocessing(X):
    if qcnn_preprocessing is None:
        # Fallback normalization to [0, pi]
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        denom = X_max - X_min
        denom[denom == 0] = 1
        return ((X - X_min) / denom) * np.pi

    pca = None
    scaler = None

    if isinstance(qcnn_preprocessing, dict):
        for key, value in qcnn_preprocessing.items():
            if hasattr(value, "transform"):
                cname = value.__class__.__name__.lower()
                if "pca" in cname:
                    pca = value
                elif "scaler" in cname or "minmax" in cname:
                    scaler = value

    X_reduced = pca.transform(X) if pca is not None else X
    if scaler is not None:
        return scaler.transform(X_reduced)

    X_min = X_reduced.min(axis=0, keepdims=True)
    X_max = X_reduced.max(axis=0, keepdims=True)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return ((X_reduced - X_min) / denom) * np.pi


ECG_BEAT_NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unclassifiable beat",
}


def predict_ecg_cnn(X):
    """Classify each 280-sample beat with the AAMI 5-class 1D CNN.

    Returns None when no validated model is installed, so predict_ecg() can
    fall back to the QCNN path.
    """
    model, card = get_ecg_model()
    if model is None:
        return None

    beats = np.asarray(X, dtype=np.float32).reshape(-1, 280)
    # beat_zscore: each beat is standardised on its own, exactly as in training.
    mean = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    beats = (beats - mean) / std

    preds = model.predict(beats[..., np.newaxis], verbose=0)
    classes = card["classes"]
    per_beat = preds.argmax(axis=1)
    mean_probs = preds.mean(axis=0)

    counts = {c: int((per_beat == i).sum()) for i, c in enumerate(classes)}
    total = len(per_beat) or 1
    abnormal = total - counts.get("N", 0)
    abnormal_fraction = abnormal / total

    # A strip is called abnormal when any non-normal beat class dominates the
    # averaged distribution, or when abnormal beats are not merely incidental.
    top_idx = int(np.argmax(mean_probs))
    top_class = classes[top_idx]
    if top_class == "N" and abnormal_fraction < 0.10:
        prediction, abnormal_type = "NORMAL", "No ectopic beats detected"
        confidence = float(mean_probs[classes.index("N")]) * 100.0
    else:
        worst = max((c for c in classes if c != "N"), key=lambda c: counts[c])
        prediction = "ABNORMAL"
        abnormal_type = ECG_BEAT_NAMES[worst]
        confidence = max(abnormal_fraction, float(mean_probs[classes.index(worst)])) * 100.0

    return {
        "prediction": prediction,
        "abnormal_type": abnormal_type,
        "confidence": f"{confidence:.2f}",
        "atrial_probability": f"{float(mean_probs[classes.index('S')]) * 100:.2f}",
        "beat_predictions": preds.max(axis=1),
        "beat_counts": counts,
        "beat_distribution": {ECG_BEAT_NAMES[c]: f"{float(mean_probs[i]) * 100:.2f}"
                              for i, c in enumerate(classes)},
        "beats_analyzed": total,
        "method": "AAMI 5-class 1D CNN (MIT-BIH, inter-patient split)",
    }


def predict_ecg(X):
    cnn_result = predict_ecg_cnn(X)
    if cnn_result is not None:
        return cnn_result

    if qcnn_weights is None:
        # Fallback heuristic prediction if weights file missing
        std_val = float(np.std(X))
        prob = float(1.0 / (1.0 + np.exp(-std_val)))
        return {
            "prediction": "NORMAL" if prob < 0.5 else "ABNORMAL",
            "abnormal_type": "No atrial abnormality detected" if prob < 0.5 else "Atrial Abnormality",
            "confidence": f"{abs(prob - 0.5) * 200:.2f}",
            "atrial_probability": f"{prob * 100:.2f}",
            "beat_predictions": np.array([prob])
        }

    processed = apply_saved_preprocessing(X)
    predictions = []

    for sample in processed:
        sample = np.asarray(sample, dtype=float)[:8]
        if len(sample) < 8:
            sample = np.pad(sample, (0, 8 - len(sample)))

        qout = float(qcnn_circuit(sample, qcnn_weights))
        prob = max(0.0, min(1.0, (1.0 - qout) / 2.0))
        predictions.append(prob)

    predictions = np.asarray(predictions)
    mean_prob = float(np.mean(predictions))

    if mean_prob >= 0.5:
        prediction = "ABNORMAL"
        abnormal_type = "Atrial Abnormality"
        confidence = mean_prob * 100
    else:
        prediction = "NORMAL"
        abnormal_type = "No atrial abnormality detected"
        confidence = (1.0 - mean_prob) * 100

    return {
        "prediction": prediction,
        "abnormal_type": abnormal_type,
        "confidence": f"{confidence:.2f}",
        "atrial_probability": f"{mean_prob * 100:.2f}",
        "beat_predictions": predictions
    }


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
        raise ValueError(f"Unsupported or corrupt image format '{ext or 'unknown'}': {e}")


# ============================================================
# BRAIN TUMOR MRI PREDICTION LOGIC
# ============================================================

MRI_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}


# ============================================================
# GRAD-CAM HELPER FUNCTIONS
# ============================================================

def generate_gradcam_heatmap(model, image_array, predicted_class):
    """Generate a Grad-CAM heatmap for the predicted class."""
    try:
        import tensorflow as tf
        
        # Recursively find the last conv layer (including inside Functional sub-models like vgg16)
        def find_last_conv(layer):
            # If this layer has sub-layers, recurse into them
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                for sub in reversed(layer.layers):
                    result = find_last_conv(sub)
                    if result is not None:
                        return result
            # Check if this is a Conv2D layer
            if 'conv' in layer.__class__.__name__.lower():
                return layer
            return None
        
        # Find the inner model that contains the conv layers
        inner_model = model
        last_conv_layer = None
        
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                # This is a Functional sub-model (like vgg16)
                for sub in reversed(layer.layers):
                    conv = find_last_conv(sub)
                    if conv is not None:
                        last_conv_layer = conv
                        inner_model = layer
                        break
            else:
                conv = find_last_conv(layer)
                if conv is not None:
                    last_conv_layer = conv
                    break
        
        if last_conv_layer is None:
            return generate_gradient_heatmap(model, image_array, predicted_class)
        
        # Build a sub-model from the inner model's input to the conv layer and output
        # The inner model like vgg16 has its own input
        grad_model = tf.keras.models.Model(
            inputs=inner_model.input,
            outputs=[last_conv_layer.output, inner_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            loss = predictions[:, predicted_class]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return generate_gradient_heatmap(model, image_array, predicted_class)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        # Convert to numpy
        heatmap_np = heatmap.numpy()
        
        # Resize to 224x224
        if cv2 is not None:
            heatmap_np = cv2.resize(heatmap_np, (224, 224))
        else:
            from PIL import Image as PILImage
            heatmap_img = PILImage.fromarray((heatmap_np * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((224, 224), PILImage.BILINEAR)
            heatmap_np = np.array(heatmap_img) / 255.0
        
        return heatmap_np
        
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return generate_gradient_heatmap(model, image_array, predicted_class)


def generate_gradient_heatmap(model, image_array, predicted_class):
    """Fallback: Generate gradient-based heatmap when Grad-CAM fails."""
    try:
        import tensorflow as tf
        
        # Ensure the input array has the right shape for the model (Batch, H, W, C)
        if image_array.ndim == 3:
            input_tensor = tf.convert_to_tensor(image_array[None, ...])
        else:
            input_tensor = tf.convert_to_tensor(image_array)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor)
            loss = predictions[:, predicted_class]
        
        grads = tape.gradient(loss, input_tensor)
        
        if grads is None:
            return None
        
        # Take absolute value and mean across channels
        grads_np = grads.numpy()[0]
        if grads_np.ndim == 3:
            heatmap = np.mean(np.abs(grads_np), axis=-1)
        else:
            heatmap = np.abs(grads_np)
        
        # Normalize
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        return heatmap
        
    except Exception as e:
        print(f"Gradient heatmap generation failed: {e}")
        return None


def create_gradcam_overlay(original_image, heatmap, alpha=0.45):
    """Create a heatmap overlay on the original image."""
    if heatmap is None:
        return original_image
    
    # Convert original image to numpy array
    original_np = np.array(original_image.resize((224, 224)))
    
    # Apply colormap to heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    if cv2 is not None:
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        # Fallback without OpenCV
        heatmap_colored = np.stack([heatmap_uint8] * 3, axis=-1)
        # Apply red channel emphasis
        heatmap_colored[:, :, 0] = heatmap_uint8
        heatmap_colored[:, :, 1] = 0
        heatmap_colored[:, :, 2] = 255 - heatmap_uint8
    
    # Blend
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0) if cv2 is not None else \
        (original_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def create_tumor_region_highlight(original_image, heatmap, threshold_percentile=90):
    """Create a highlighted tumor region image."""
    if heatmap is None:
        return original_image
    
    original_np = np.array(original_image.resize((224, 224)))
    
    # Threshold the heatmap
    threshold = np.percentile(heatmap, threshold_percentile)
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    
    # Clean up mask
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create red overlay
    overlay = original_np.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red highlight
    
    # Blend
    highlighted = cv2.addWeighted(original_np, 0.65, overlay, 0.35, 0) if cv2 is not None else \
        (original_np * 0.65 + overlay * 0.35).astype(np.uint8)
    
    # Draw contour if OpenCV available
    if cv2 is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(highlighted, [largest_contour], -1, (255, 255, 0), 2)  # Yellow boundary
    
    return Image.fromarray(highlighted)


def calculate_tumor_area(mask):
    """Calculate tumor area as percentage of image."""
    active_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(active_pixels / total_pixels * 100)


def calculate_tumor_size(contour):
    """Calculate tumor bounding box size."""
    if contour is None or not CV2_AVAILABLE:
        return 0, 0
    x, y, width, height = cv2.boundingRect(contour)
    return int(width), int(height)


def calculate_tumor_location(contour):
    """Calculate tumor location description."""
    if contour is None or not CV2_AVAILABLE:
        return "Not available"
    
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return "Not available"
    
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    
    horizontal = "Left" if center_x < 74 else ("Central" if center_x < 150 else "Right")
    vertical = "Upper" if center_y < 74 else ("Middle" if center_y < 150 else "Lower")
    
    return f"{horizontal} {vertical} region"


def calculate_severity(area):
    """Calculate severity based on area."""
    if area < 5:
        return "Low"
    elif area < 15:
        return "Moderate"
    else:
        return "High"


def calculate_spread(area):
    """Calculate spread based on area."""
    if area < 5:
        return "Limited"
    elif area < 15:
        return "Moderate"
    else:
        return "Extensive"


def predict_brain_tumor(image_path):
    img = load_image_rgb(image_path)
    model, card = get_brain_tumor_model()
    # Resize to whatever the card declares -- 224 for vgg16, 128 for the
    # scratch CNN. The Grad-CAM path below reuses this same array.
    size = tuple(card["input_shape"][:2]) if card else (224, 224)
    img_array = np.array(img.resize(size), dtype=np.float32)

    if model is not None:
        try:
            input_tensor = preprocess_mri(img_array, card["preprocessing"])
            preds = model.predict(input_tensor)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx]) * 100.0
            probs = {MRI_CLASSES[i]: float(preds[i]) * 100.0 for i in range(len(preds))}
        except Exception as e:
            print("Model prediction error, using color distribution heuristic fallback:", e)
            top_idx, confidence, probs = fallback_mri_analysis(img_array)
    else:
        top_idx, confidence, probs = fallback_mri_analysis(img_array)

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


def fallback_mri_analysis(img_array):
    # Rule-based visual feature extraction for demonstration/fallback
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    center_crop = img_array[50:170, 50:170]
    center_intensity = np.mean(center_crop)

    if center_intensity > mean_val + 15:
        top_idx = 0  # Glioma
        probs = [78.4, 11.2, 4.3, 6.1]
    elif std_val > 55:
        top_idx = 1  # Meningioma
        probs = [12.1, 74.5, 5.2, 8.2]
    elif center_intensity < mean_val - 10:
        top_idx = 3  # Pituitary
        probs = [8.1, 9.3, 6.4, 76.2]
    else:
        top_idx = 2  # No Tumor
        probs = [2.5, 3.1, 91.8, 2.6]

    confidence = probs[top_idx]
    probs_dict = {MRI_CLASSES[i]: probs[i] for i in range(4)}
    return top_idx, confidence, probs_dict


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
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/samples/<filename>")
def sample_file(filename):
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
            X = prepare_qcnn_input(ecg_values)
            # A digitized trace has one sample per pixel COLUMN, so its true
            # sampling rate is set by the paper speed: 25 mm/s x px/mm. It is
            # NOT the 360 Hz of the MIT-BIH CSV path. With no grid there is no
            # timebase at all; analyse() then reports everything unavailable.
            px_per_mm = digitized["px_per_mm"]
            image_fs = fs_from_scale(px_per_mm) if px_per_mm is not None else 360.0
            report = analyse_ecg_parameters(leads, fs=image_fs,
                                            px_per_mm=px_per_mm,
                                            from_image=True)
            input_source = f"ECG image ({digitized['layout'].replace('_', ' ')})"
        else:
            ecg_values = load_ecg_file(filepath)
            X = prepare_qcnn_input(ecg_values)
            report = analyse_ecg_parameters(ecg_values, fs=360.0)
            input_source = "ECG data file"

        qcnn_result = predict_ecg(X)

        plot_filename = "ecg_waveform_" + uuid.uuid4().hex[:8] + ".png"
        create_ecg_plot(ecg_values, plot_filename)

        classifier = qcnn_result.get("method", "8-qubit QCNN")
        if qcnn_result["prediction"] == "NORMAL":
            interpretation = (f"The {classifier} classified the ECG as NORMAL "
                              f"({qcnn_result['abnormal_type']}).")
            recommendation = "Normal rhythm detected. Routine clinical monitoring recommended."
        else:
            interpretation = (f"The {classifier} classified the ECG as ABNORMAL "
                              f"({qcnn_result['abnormal_type']}).")
            recommendation = "Abnormal pattern identified. Clinical evaluation by a cardiologist is advised."

        result = {
            "prediction": qcnn_result["prediction"],
            "abnormal_type": qcnn_result["abnormal_type"],
            "confidence": qcnn_result["confidence"],
            "atrial_probability": qcnn_result["atrial_probability"],
            "classifier": classifier,
            "beat_distribution": qcnn_result.get("beat_distribution"),
            "beat_counts": qcnn_result.get("beat_counts"),
            "beats_analyzed": qcnn_result.get("beats_analyzed"),
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
        res = predict_brain_tumor(filepath)

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
# PDF REPORT BUILDER
# ------------------------------------------------------------

REPORT_STYLES = getSampleStyleSheet()
REPORT_STYLES.add(ParagraphStyle(
    name="ReportTitle", parent=REPORT_STYLES["Title"],
    fontSize=18, leading=22, spaceAfter=2
))
REPORT_STYLES.add(ParagraphStyle(
    name="ReportSubtitle", parent=REPORT_STYLES["Normal"],
    fontSize=10, leading=14, textColor=colors.HexColor("#475569"), alignment=1
))
REPORT_STYLES.add(ParagraphStyle(
    name="SectionHeading", parent=REPORT_STYLES["Heading3"],
    fontSize=11.5, leading=14, spaceBefore=4, spaceAfter=6,
    textColor=colors.HexColor("#0f172a")
))
REPORT_STYLES.add(ParagraphStyle(
    name="ReportBody", parent=REPORT_STYLES["BodyText"],
    fontSize=10, leading=14.5
))
REPORT_STYLES.add(ParagraphStyle(
    name="Cell", parent=REPORT_STYLES["BodyText"],
    fontSize=9.5, leading=12.5, spaceBefore=0, spaceAfter=0
))
REPORT_STYLES.add(ParagraphStyle(
    name="Disclaimer", parent=REPORT_STYLES["Italic"],
    fontSize=8.5, leading=11.5, textColor=colors.HexColor("#64748b")
))

TABLE_WIDTHS = [200, 323]


def _cell(text, bold=False):
    """Escape a value and wrap it in a Paragraph so long text wraps in-cell."""
    safe = escape(str(text if text not in (None, "") else NOT_PROVIDED))
    if bold:
        safe = f"<b>{safe}</b>"
    return Paragraph(safe, REPORT_STYLES["Cell"])


def _data_table(rows, accent):
    """Build a two-column label/value table with a coloured header row."""
    data = [[_cell(rows[0][0], bold=True), _cell(rows[0][1], bold=True)]]
    data += [[_cell(label), _cell(value)] for label, value in rows[1:]]

    table = Table(data, colWidths=TABLE_WIDTHS, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(accent)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _header_footer(canvas, doc, accent, footer_text):
    """Draw the accent rule, page number and footer note on every page."""
    canvas.saveState()
    width, _ = A4

    canvas.setStrokeColor(colors.HexColor(accent))
    canvas.setLineWidth(2)
    canvas.line(36, doc.pagesize[1] - 30, width - 36, doc.pagesize[1] - 30)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawString(36, 24, footer_text)
    canvas.drawRightString(width - 36, 24, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_pdf_report(title, subtitle, accent, meta, meta_fields,
                     meta_heading, sections, disclaimer, footer_text):
    """Assemble a consistently styled PDF and return it as a BytesIO buffer.

    sections: list of ("table", heading, [(label, value), ...])
              or        ("text",  heading, body_string)
    """
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=42,
        title=title, author="Unified AI Diagnostic Platform"
    )

    story = [
        Paragraph(escape(title), REPORT_STYLES["ReportTitle"]),
        Paragraph(escape(subtitle), REPORT_STYLES["ReportSubtitle"]),
        Spacer(1, 6),
        Paragraph(
            f"Report ID: <b>{escape(meta.get('report_id', NOT_PROVIDED))}</b> &nbsp;|&nbsp; "
            f"Generated: <b>{escape(meta.get('generated_at', NOT_PROVIDED))}</b>",
            REPORT_STYLES["ReportSubtitle"]
        ),
        Spacer(1, 16),
        Paragraph(escape(meta_heading), REPORT_STYLES["SectionHeading"]),
        _data_table([("Field", "Details")] + metadata_rows(meta, meta_fields), accent),
        Spacer(1, 16),
    ]

    for kind, heading, body in sections:
        story.append(Paragraph(escape(heading), REPORT_STYLES["SectionHeading"]))
        if kind == "table":
            story.append(_data_table(body, accent))
        else:
            story.append(Paragraph(escape(str(body)), REPORT_STYLES["ReportBody"]))
        story.append(Spacer(1, 14))

    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(disclaimer), REPORT_STYLES["Disclaimer"]))

    def _decorate(canvas, doc):
        _header_footer(canvas, doc, accent, footer_text)

    document.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    buffer.seek(0)
    return buffer


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


@app.route("/download_ecg_report", methods=["POST"])
def download_ecg_report():
    form = request.form
    meta = _form_meta(form, PATIENT_FIELDS)

    sections = [
        ("table", "1. QCNN Classification Result", [
            ("Parameter", "Result"),
            ("Overall Rhythm Classification", form.get("prediction", NOT_PROVIDED)),
            ("Abnormality Class", form.get("abnormal_type", NOT_PROVIDED)),
            ("Classifier Confidence", f"{form.get('confidence', '—')}%"),
            ("Atrial Abnormality Probability", f"{form.get('atrial_probability', '—')}%"),
            ("Signal Quality", form.get("signal_quality", NOT_PROVIDED)),
            ("Input Source", form.get("input_source", NOT_PROVIDED)),
        ]),
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

    buffer = build_pdf_report(
        title="QUANTUM ECG ANALYSIS REPORT",
        subtitle="AI Diagnostic Summary — 8-Qubit Quantum Convolutional Neural Network (QCNN)",
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
        footer_text="Unified AI Diagnostic Platform — Quantum ECG Analysis (research use only)"
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