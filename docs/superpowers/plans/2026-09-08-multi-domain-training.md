# Multi-Domain Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train CNN and VGG16 models for brain MRI, a CNN for EuroSAT land cover, and a 1D CNN for MIT-BIH ECG, from professionally organized datasets with a model registry that prevents preprocessing and task mismatches.

**Architecture:** A shared `mlkit` package holds the registry, manifest, and split utilities. One preparation script per domain writes `data/processed/` plus a checksummed manifest. One training script per model writes a `.keras` file and a JSON sidecar. The Flask app loads models only via the registry, which validates the sidecar against the caller's expectations.

**Tech Stack:** TensorFlow 2.21 (Keras 3), scikit-learn 1.9, wfdb 4.3, pyarrow, NumPy 2, Flask, pytest.

## Global Constraints

- Python interpreter is `.venv/bin/python`. Install packages with `VIRTUAL_ENV=.venv uv pip install <pkg>` — the venv has no `pip`.
- CPU-only: 4 cores, ~3 GB free RAM. Never load a full image dataset into memory; use generators or batched writes.
- Every split is deterministic. `SEED = 42` everywhere; pass it to every shuffle and split.
- ECG splits are by record (patient), never by beat. Paced records 102, 104, 107, 217 are excluded.
- Every trained model writes a sidecar `<model>.json` next to the `.keras` file.
- Class order in a sidecar is the authoritative label order; never re-derive it from a directory listing at inference.
- Tests live in `tests/` and run with `.venv/bin/python -m pytest`.
- Commit after every task.

---

### Task 1: Model registry

**Files:**
- Create: `mlkit/__init__.py`, `mlkit/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `save_model_card(path, task, classes, input_shape, preprocessing, metrics) -> dict`, `load_card(model_path) -> dict`, `validate_card(card, task, classes, input_shape, preprocessing) -> None` (raises `RegistryError`), `RegistryError`.

- [ ] **Step 1: Write the failing test**

```python
import json, pytest
from mlkit.registry import save_model_card, load_card, validate_card, RegistryError

def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "m.keras"; p.write_bytes(b"x")
    save_model_card(str(p), "mri", ["a", "b"], [8, 8, 3], "rescale_255", {"test_accuracy": 0.9})
    card = load_card(str(p))
    assert card["task"] == "mri"
    assert card["classes"] == ["a", "b"]
    assert card["preprocessing"] == "rescale_255"

def test_validate_rejects_wrong_task(tmp_path):
    card = {"task": "mri", "classes": ["a"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="task"):
        validate_card(card, "satellite", ["a"], [8, 8, 3], "rescale_255")

def test_validate_rejects_class_mismatch(tmp_path):
    card = {"task": "mri", "classes": ["a", "b"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="class"):
        validate_card(card, "mri", ["a", "b", "c"], [8, 8, 3], "rescale_255")

def test_validate_rejects_preprocessing_mismatch():
    card = {"task": "mri", "classes": ["a"], "input_shape": [8, 8, 3], "preprocessing": "rescale_255"}
    with pytest.raises(RegistryError, match="preprocessing"):
        validate_card(card, "mri", ["a"], [8, 8, 3], "vgg16_preprocess_input")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlkit'`

- [ ] **Step 3: Write minimal implementation**

```python
# mlkit/registry.py
"""Model cards: what a model was trained on and how its inputs are prepared.

Both severe defects found in this codebase -- a model fed /255 inputs when it
was trained with vgg16 preprocess_input, and a brain-tumor model serving as a
land-cover classifier -- were undetectable because nothing recorded these facts.
"""
import json, os
from datetime import date


class RegistryError(RuntimeError):
    """A model does not match what the calling code expects."""


def card_path(model_path):
    return os.path.splitext(model_path)[0] + ".json"


def save_model_card(model_path, task, classes, input_shape, preprocessing, metrics):
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
    p = card_path(model_path)
    if not os.path.exists(p):
        raise RegistryError(f"No model card beside {model_path}; refusing to load an undocumented model.")
    with open(p) as f:
        return json.load(f)


def validate_card(card, task, classes, input_shape, preprocessing):
    if card.get("task") != task:
        raise RegistryError(f"task mismatch: card says {card.get('task')!r}, caller expects {task!r}")
    if list(card.get("classes", [])) != list(classes):
        raise RegistryError(f"class mismatch: card has {card.get('classes')}, caller expects {list(classes)}")
    if list(card.get("input_shape", [])) != list(input_shape):
        raise RegistryError(f"input_shape mismatch: card {card.get('input_shape')} vs {list(input_shape)}")
    if card.get("preprocessing") != preprocessing:
        raise RegistryError(f"preprocessing mismatch: card {card.get('preprocessing')!r} vs {preprocessing!r}")
```

Also create empty `mlkit/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlkit tests/test_registry.py && git commit -m "feat: add model registry with card validation"
```

---

### Task 2: Manifest utilities

**Files:**
- Create: `mlkit/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sha256(path) -> str`, `write_manifest(rows, out_csv) -> None` where rows are dicts with keys `path,label,split,sha256,source`, `read_manifest(csv) -> list[dict]`, `assert_no_split_overlap(rows, key) -> None` (raises `LeakageError`), `LeakageError`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from mlkit.manifest import sha256, write_manifest, read_manifest, assert_no_split_overlap, LeakageError

def test_sha256_stable(tmp_path):
    f = tmp_path / "a.txt"; f.write_bytes(b"hello")
    assert sha256(str(f)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

def test_manifest_roundtrip(tmp_path):
    rows = [{"path": "a.png", "label": "x", "split": "train", "sha256": "d", "source": "s"}]
    out = tmp_path / "m.csv"
    write_manifest(rows, str(out))
    assert read_manifest(str(out)) == rows

def test_overlap_detected():
    rows = [{"sha256": "d", "split": "train"}, {"sha256": "d", "split": "test"}]
    with pytest.raises(LeakageError):
        assert_no_split_overlap(rows, "sha256")

def test_no_overlap_passes():
    rows = [{"sha256": "a", "split": "train"}, {"sha256": "b", "split": "test"}]
    assert_no_split_overlap(rows, "sha256")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: FAIL, module missing

- [ ] **Step 3: Implement**

```python
# mlkit/manifest.py
"""Dataset manifests: every processed file's checksum, label and split."""
import csv, hashlib
from collections import defaultdict

FIELDS = ["path", "label", "split", "sha256", "source"]


class LeakageError(RuntimeError):
    """The same item appears in more than one split."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assert_no_split_overlap(rows, key):
    seen = defaultdict(set)
    for r in rows:
        seen[r[key]].add(r["split"])
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    if bad:
        sample = list(bad.items())[:5]
        raise LeakageError(f"{len(bad)} item(s) appear in multiple splits, e.g. {sample}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mlkit/manifest.py tests/test_manifest.py && git commit -m "feat: add dataset manifest and leakage detection"
```

---

### Task 3: Prepare the MRI dataset

**Files:**
- Create: `scripts/prepare_mri.py`
- Test: `tests/test_prepare_mri.py`

**Interfaces:**
- Consumes: `mlkit.manifest`.
- Produces: `data/processed/mri/{train,val,test}/<class>/*.jpg`, `data/manifests/mri.csv`. Classes: `["glioma","meningioma","notumor","pituitary"]`.

- [ ] **Step 1: Write the script**

```python
# scripts/prepare_mri.py
"""Build data/processed/mri from the on-disk dataset/ tree.

dataset/Testing is the held-out test set. A stratified 15% validation split is
carved from dataset/Training with a fixed seed.
"""
import os, random, shutil, sys
sys.path.insert(0, os.getcwd())
from mlkit.manifest import sha256, write_manifest, assert_no_split_overlap

SEED = 42
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SRC, OUT = "dataset", "data/processed/mri"
VAL_FRACTION = 0.15


def main():
    random.seed(SEED)
    rows = []
    for split_src, split_out in [("Training", None), ("Testing", "test")]:
        for cls in CLASSES:
            files = sorted(os.listdir(os.path.join(SRC, split_src, cls)))
            if split_out is None:
                random.shuffle(files)
                n_val = int(len(files) * VAL_FRACTION)
                assignment = [("val", f) for f in files[:n_val]] + [("train", f) for f in files[n_val:]]
            else:
                assignment = [(split_out, f) for f in files]
            for split, fname in assignment:
                dst_dir = os.path.join(OUT, split, cls)
                os.makedirs(dst_dir, exist_ok=True)
                src_p = os.path.join(SRC, split_src, cls, fname)
                dst_p = os.path.join(dst_dir, fname)
                if not os.path.exists(dst_p):
                    shutil.copy2(src_p, dst_p)
                rows.append({"path": dst_p, "label": cls, "split": split,
                             "sha256": sha256(dst_p), "source": f"{SRC}/{split_src}/{cls}/{fname}"})
    assert_no_split_overlap(rows, "sha256")
    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/mri.csv")
    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {}).setdefault(r["label"], 0)
        counts[r["split"]][r["label"]] += 1
    print("MRI prepared:", counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/prepare_mri.py`
Expected: prints per-split counts; train ~4760, val ~840, test 1600.

- [ ] **Step 3: Write the verification test**

```python
import csv, os, pytest
from mlkit.manifest import read_manifest, assert_no_split_overlap, sha256

MANIFEST = "data/manifests/mri.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run scripts/prepare_mri.py first")

def test_no_leakage_between_splits():
    assert_no_split_overlap(read_manifest(MANIFEST), "sha256")

def test_all_splits_present_and_balanced():
    rows = read_manifest(MANIFEST)
    splits = {r["split"] for r in rows}
    assert splits == {"train", "val", "test"}
    for split in splits:
        labels = {r["label"] for r in rows if r["split"] == split}
        assert labels == {"glioma", "meningioma", "notumor", "pituitary"}

def test_checksums_match_files_on_disk():
    rows = read_manifest(MANIFEST)[:50]
    for r in rows:
        assert os.path.exists(r["path"])
        assert sha256(r["path"]) == r["sha256"]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_mri.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_mri.py tests/test_prepare_mri.py && git commit -m "feat: prepare MRI dataset with manifest and leakage check"
```

---

### Task 4: Train the MRI scratch CNN

**Files:**
- Create: `scripts/train_mri_cnn.py`
- Produces: `model/mri_cnn.keras` + `model/mri_cnn.json`

**Interfaces:**
- Consumes: `data/processed/mri/`, `mlkit.registry.save_model_card`.
- Produces: model card with `task="mri"`, `input_shape=[128,128,3]`, `preprocessing="rescale_255"`.

- [ ] **Step 1: Write the training script**

```python
# scripts/train_mri_cnn.py
"""Scratch CNN for 4-class brain MRI classification at 128x128."""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from mlkit.registry import save_model_card

SEED, SIZE, BATCH, EPOCHS = 42, 128, 32, 30
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
tf.keras.utils.set_random_seed(SEED)


def loader(split, shuffle, augment=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"data/processed/mri/{split}", labels="inferred", label_mode="categorical",
        class_names=CLASSES, image_size=(SIZE, SIZE), batch_size=BATCH,
        shuffle=shuffle, seed=SEED)
    rescale = layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"), layers.RandomRotation(0.04),
            layers.RandomZoom(0.1), layers.RandomTranslation(0.1, 0.1)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build():
    m = models.Sequential([layers.Input((SIZE, SIZE, 3))])
    for f in [32, 64, 128, 128]:
        m.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization()); m.add(layers.ReLU())
        m.add(layers.MaxPooling2D())
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(len(CLASSES), activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    train, val, test = loader("train", True, True), loader("val", False), loader("test", False)
    model = build()
    model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])
    y_true = np.concatenate([np.argmax(y, 1) for _, y in test])
    y_pred = np.argmax(model.predict(test), 1)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("confusion:\n", confusion_matrix(y_true, y_pred))
    os.makedirs("model", exist_ok=True)
    path = "model/mri_cnn.keras"
    model.save(path)
    save_model_card(path, "mri", CLASSES, [SIZE, SIZE, 3], "rescale_255",
                    {"test_accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"]})
    print("saved", path, "acc", rep["accuracy"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train**

Run: `.venv/bin/python scripts/train_mri_cnn.py 2>&1 | tail -30`
Expected: test accuracy >= 0.88; `model/mri_cnn.keras` and `.json` written.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_mri_cnn.py model/mri_cnn.json && git commit -m "feat: train scratch CNN for MRI classification"
```

---

### Task 5: Train the MRI VGG16 transfer model

**Files:**
- Create: `scripts/train_mri_vgg16.py`
- Produces: `model/mri_vgg16.keras` + `model/mri_vgg16.json`

**Interfaces:**
- Produces: model card with `task="mri"`, `input_shape=[224,224,3]`, `preprocessing="vgg16_preprocess_input"`.

- [ ] **Step 1: Write the training script**

Bottleneck features are cached to `data/processed/mri_vgg16_features/` so the
head trains in seconds per epoch on CPU.

```python
# scripts/train_mri_vgg16.py
"""VGG16 transfer learning for brain MRI, with cached bottleneck features."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from mlkit.registry import save_model_card

SEED, SIZE, BATCH = 42, 224, 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CACHE = "data/processed/mri_vgg16_features"
tf.keras.utils.set_random_seed(SEED)


def raw_loader(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"data/processed/mri/{split}", labels="inferred", label_mode="categorical",
        class_names=CLASSES, image_size=(SIZE, SIZE), batch_size=BATCH, shuffle=False)
    return ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)


def features(split, base):
    os.makedirs(CACHE, exist_ok=True)
    fx, fy = f"{CACHE}/{split}_x.npy", f"{CACHE}/{split}_y.npy"
    if os.path.exists(fx):
        return np.load(fx), np.load(fy)
    xs, ys = [], []
    for i, (bx, by) in enumerate(raw_loader(split)):
        xs.append(base.predict(bx, verbose=0)); ys.append(by.numpy())
        if i % 20 == 0:
            print(f"  {split} batch {i}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.save(fx, x); np.save(fy, y)
    return x, y


def main():
    base = VGG16(include_top=False, weights="imagenet",
                 input_shape=(SIZE, SIZE, 3), pooling="avg")
    base.trainable = False
    xtr, ytr = features("train", base)
    xva, yva = features("val", base)
    xte, yte = features("test", base)
    head = models.Sequential([
        layers.Input((xtr.shape[1],)),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation="softmax")])
    head.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    head.fit(xtr, ytr, validation_data=(xva, yva), epochs=60, batch_size=64, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)])
    y_true, y_pred = np.argmax(yte, 1), np.argmax(head.predict(xte, verbose=0), 1)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("confusion:\n", confusion_matrix(y_true, y_pred))
    full = models.Sequential([layers.Input((SIZE, SIZE, 3)), base, head])
    path = "model/mri_vgg16.keras"
    full.save(path)
    save_model_card(path, "mri", CLASSES, [SIZE, SIZE, 3], "vgg16_preprocess_input",
                    {"test_accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"]})
    print("saved", path, "acc", rep["accuracy"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train**

Run: `.venv/bin/python scripts/train_mri_vgg16.py 2>&1 | tail -30`
Expected: test accuracy >= 0.93. Feature extraction is the slow part (~45-60 min).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_mri_vgg16.py model/mri_vgg16.json && git commit -m "feat: train VGG16 transfer model for MRI"
```

---

### Task 6: Prepare and train EuroSAT

**Files:**
- Create: `scripts/prepare_eurosat.py`, `scripts/train_eurosat.py`
- Test: `tests/test_prepare_eurosat.py`
- Produces: `model/satellite_best.keras` + `model/satellite_best.json`

**Interfaces:**
- Produces: card `task="satellite"`, `input_shape=[64,64,3]`, `preprocessing="rescale_255"`, 10 classes in EuroSAT label order.

- [ ] **Step 1: Write the preparation script**

```python
# scripts/prepare_eurosat.py
"""Extract EuroSAT parquet shards into an image tree plus a manifest."""
import io, os, sys
sys.path.insert(0, os.getcwd())
import pyarrow.parquet as pq
from PIL import Image
from mlkit.manifest import sha256, write_manifest, assert_no_split_overlap

RAW, OUT = "data/raw/eurosat", "data/processed/eurosat"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def main():
    rows = []
    for src_split, out_split in SPLITS.items():
        table = pq.read_table(f"{RAW}/{src_split}.parquet")
        cols = table.column_names
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        names = table.column("filename").to_pylist() if "filename" in cols else None
        for i, (img, lab) in enumerate(zip(images, labels)):
            cls = CLASSES[lab]
            d = os.path.join(OUT, out_split, cls)
            os.makedirs(d, exist_ok=True)
            fname = (names[i] if names else f"{out_split}_{i}.jpg")
            fname = os.path.basename(fname)
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fname += ".jpg"
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(p, quality=95)
            rows.append({"path": p, "label": cls, "split": out_split,
                         "sha256": sha256(p), "source": f"EuroSAT_RGB/{src_split}"})
        print(f"  {src_split}: {len(images)} images", flush=True)
    assert_no_split_overlap(rows, "sha256")
    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/eurosat.csv")
    print("EuroSAT prepared:", len(rows), "images")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/prepare_eurosat.py`
Expected: ~27,000 images across train/val/test, manifest written.

- [ ] **Step 3: Write the training script**

```python
# scripts/train_eurosat.py
"""Scratch CNN for 10-class EuroSAT land cover at native 64x64."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from mlkit.registry import save_model_card

SEED, SIZE, BATCH, EPOCHS = 42, 64, 64, 40
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
tf.keras.utils.set_random_seed(SEED)


def loader(split, shuffle, augment=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"data/processed/eurosat/{split}", labels="inferred", label_mode="categorical",
        class_names=CLASSES, image_size=(SIZE, SIZE), batch_size=BATCH,
        shuffle=shuffle, seed=SEED)
    rescale = layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                   layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build():
    m = models.Sequential([layers.Input((SIZE, SIZE, 3))])
    for f in [32, 64, 128, 256]:
        m.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization()); m.add(layers.ReLU())
        m.add(layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(layers.BatchNormalization()); m.add(layers.ReLU())
        m.add(layers.MaxPooling2D())
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(len(CLASSES), activation="softmax"))
    m.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    train, val, test = loader("train", True, True), loader("val", False), loader("test", False)
    model = build()
    model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])
    y_true = np.concatenate([np.argmax(y, 1) for _, y in test])
    y_pred = np.argmax(model.predict(test), 1)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("confusion:\n", confusion_matrix(y_true, y_pred))
    path = "model/satellite_best.keras"
    model.save(path)
    save_model_card(path, "satellite", CLASSES, [SIZE, SIZE, 3], "rescale_255",
                    {"test_accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"]})
    print("saved", path, "acc", rep["accuracy"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Train**

Run: `.venv/bin/python scripts/train_eurosat.py 2>&1 | tail -30`
Expected: test accuracy >= 0.88.

- [ ] **Step 5: Write the leakage test**

```python
import os, pytest
from mlkit.manifest import read_manifest, assert_no_split_overlap

MANIFEST = "data/manifests/eurosat.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run scripts/prepare_eurosat.py first")

def test_no_leakage():
    assert_no_split_overlap(read_manifest(MANIFEST), "sha256")

def test_ten_classes_in_every_split():
    rows = read_manifest(MANIFEST)
    for split in ["train", "val", "test"]:
        labels = {r["label"] for r in rows if r["split"] == split}
        assert len(labels) == 10, f"{split} has {len(labels)} classes"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_eurosat.py scripts/train_eurosat.py tests/test_prepare_eurosat.py model/satellite_best.json && git commit -m "feat: prepare EuroSAT and train 10-class land cover CNN"
```

---

### Task 7: Prepare the ECG beat dataset

**Files:**
- Create: `scripts/prepare_ecg.py`
- Test: `tests/test_prepare_ecg.py`
- Produces: `data/processed/ecg/{train,val,test}.npz`, `data/manifests/ecg.csv`

**Interfaces:**
- Produces: npz files with arrays `X` shape `(n, 280)`, `y` shape `(n,)` int labels, `records` shape `(n,)` string. Classes `["N","S","V","F","Q"]`.

- [ ] **Step 1: Write the preparation script**

```python
# scripts/prepare_ecg.py
"""Segment MIT-BIH into AAMI-labelled beats with an inter-patient split.

Splitting by beat instead of by record leaks patient identity across splits and
inflates accuracy to ~99%. This uses the de Chazal DS1/DS2 record split.
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, wfdb
from mlkit.manifest import write_manifest

RAW, OUT = "data/raw/mitdb", "data/processed/ecg"
SEED, BEFORE, AFTER = 42, 100, 180          # 280 samples at 360 Hz
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in ["A", "a", "J", "S"]},
        **{s: "V" for s in ["V", "E"]}, "F": "F",
        **{s: "Q" for s in ["/", "f", "Q"]}}


def beats_for(record):
    rec = wfdb.rdrecord(os.path.join(RAW, str(record)))
    ann = wfdb.rdann(os.path.join(RAW, str(record)), "atr")
    lead = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    sig = rec.p_signal[:, lead].astype(np.float32)
    X, y = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        cls = AAMI.get(symbol)
        if cls is None:
            continue
        s, e = sample - BEFORE, sample + AFTER
        if s < 0 or e > len(sig):
            continue
        beat = sig[s:e]
        std = beat.std()
        beat = (beat - beat.mean()) / (std if std > 1e-6 else 1.0)
        X.append(beat); y.append(CLASSES.index(cls))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def main():
    splits = json.load(open(f"{RAW}/splits.json"))
    rng = np.random.default_rng(SEED)
    ds1 = list(splits["DS1"]); rng.shuffle(ds1)
    n_val = max(1, len(ds1) // 5)
    assign = {r: "val" for r in ds1[:n_val]}
    assign.update({r: "train" for r in ds1[n_val:]})
    assign.update({r: "test" for r in splits["DS2"]})

    os.makedirs(OUT, exist_ok=True)
    buckets, rows = {"train": [], "val": [], "test": []}, []
    for record, split in sorted(assign.items()):
        X, y = beats_for(record)
        if len(X) == 0:
            continue
        buckets[split].append((X, y, np.full(len(X), str(record))))
        rows.append({"path": f"{OUT}/{split}.npz", "label": "mixed", "split": split,
                     "sha256": "", "source": f"mitdb/{record}"})
        print(f"  record {record} -> {split}: {len(X)} beats", flush=True)

    for split, parts in buckets.items():
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        recs = np.concatenate([p[2] for p in parts])
        np.savez_compressed(f"{OUT}/{split}.npz", X=X, y=y, records=recs)
        dist = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
        print(f"{split}: {len(X)} beats {dist}")
    write_manifest(rows, "data/manifests/ecg.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/prepare_ecg.py`
Expected: ~50k train beats, ~50k test beats, N dominant.

- [ ] **Step 3: Write the leakage test**

```python
import os, numpy as np, pytest

OUT = "data/processed/ecg"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{OUT}/train.npz"), reason="run scripts/prepare_ecg.py first")

def _recs(split):
    return set(np.load(f"{OUT}/{split}.npz", allow_pickle=True)["records"].tolist())

def test_no_patient_appears_in_two_splits():
    tr, va, te = _recs("train"), _recs("val"), _recs("test")
    assert tr & te == set(), f"record leak train/test: {tr & te}"
    assert tr & va == set(), f"record leak train/val: {tr & va}"
    assert va & te == set(), f"record leak val/test: {va & te}"

def test_paced_records_excluded():
    all_recs = _recs("train") | _recs("val") | _recs("test")
    assert all_recs.isdisjoint({"102", "104", "107", "217"})

def test_beat_shape_is_280():
    X = np.load(f"{OUT}/train.npz")["X"]
    assert X.shape[1] == 280
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_ecg.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_ecg.py tests/test_prepare_ecg.py && git commit -m "feat: build MIT-BIH AAMI beat dataset with inter-patient split"
```

---

### Task 8: Train the ECG 1D CNN

**Files:**
- Create: `scripts/train_ecg_cnn.py`
- Produces: `model/ecg_cnn.keras` + `model/ecg_cnn.json`

**Interfaces:**
- Produces: card `task="ecg"`, `input_shape=[280,1]`, `preprocessing="beat_zscore"`, classes `["N","S","V","F","Q"]`.

- [ ] **Step 1: Write the training script**

```python
# scripts/train_ecg_cnn.py
"""1D CNN over 280-sample MIT-BIH beats, 5 AAMI classes."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from mlkit.registry import save_model_card

SEED, CLASSES = 42, ["N", "S", "V", "F", "Q"]
tf.keras.utils.set_random_seed(SEED)


def load(split):
    d = np.load(f"data/processed/ecg/{split}.npz")
    return d["X"][..., None].astype("float32"), d["y"]


def build():
    m = models.Sequential([layers.Input((280, 1))])
    for f, k in [(32, 7), (64, 5), (128, 3)]:
        m.add(layers.Conv1D(f, k, padding="same", use_bias=False))
        m.add(layers.BatchNormalization()); m.add(layers.ReLU())
        m.add(layers.MaxPooling1D(2))
    m.add(layers.GlobalAveragePooling1D())
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(len(CLASSES), activation="softmax"))
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    xtr, ytr = load("train"); xva, yva = load("val"); xte, yte = load("test")
    present = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=present, y=ytr)
    cw = {int(c): float(w) for c, w in zip(present, weights)}
    print("class weights:", cw)
    model = build()
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=40, batch_size=128,
              class_weight=cw, callbacks=[
                  tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])
    y_pred = np.argmax(model.predict(xte, verbose=0), 1)
    labels = list(range(len(CLASSES)))
    rep = classification_report(yte, y_pred, labels=labels, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    print(classification_report(yte, y_pred, labels=labels, target_names=CLASSES, zero_division=0))
    print("confusion:\n", confusion_matrix(yte, y_pred, labels=labels))
    path = "model/ecg_cnn.keras"
    model.save(path)
    save_model_card(path, "ecg", CLASSES, [280, 1], "beat_zscore",
                    {"test_accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"]})
    print("saved", path, "macro F1", rep["macro avg"]["f1-score"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train**

Run: `.venv/bin/python scripts/train_ecg_cnn.py 2>&1 | tail -30`
Expected: macro F1 >= 0.60 inter-patient.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_ecg_cnn.py model/ecg_cnn.json && git commit -m "feat: train 1D CNN for MIT-BIH AAMI beat classification"
```

---

### Task 9: Wire the registry into the app

**Files:**
- Modify: `app.py` (model loaders, `predict_satellite`, `predict_ecg`, `MRI_CLASSES`)
- Modify: `templates/satellite.html`, `templates/ecg.html`
- Test: `tests/test_model_contracts.py`

**Interfaces:**
- Consumes: `mlkit.registry.load_card`, `validate_card`; all four model cards.

- [ ] **Step 1: Write the contract test**

```python
import os, pytest, numpy as np
from mlkit.registry import load_card, validate_card

CASES = [
    ("model/mri_vgg16.keras", "mri", 4, [224, 224, 3], "vgg16_preprocess_input"),
    ("model/mri_cnn.keras", "mri", 4, [128, 128, 3], "rescale_255"),
    ("model/satellite_best.keras", "satellite", 10, [64, 64, 3], "rescale_255"),
    ("model/ecg_cnn.keras", "ecg", 5, [280, 1], "beat_zscore"),
]

@pytest.mark.parametrize("path,task,n,shape,prep", CASES)
def test_card_matches_expectations(path, task, n, shape, prep):
    if not os.path.exists(path):
        pytest.skip(f"{path} not trained yet")
    card = load_card(path)
    assert card["task"] == task
    assert len(card["classes"]) == n
    validate_card(card, task, card["classes"], shape, prep)

@pytest.mark.parametrize("path,task,n,shape,prep", CASES)
def test_model_output_matches_card(path, task, n, shape, prep):
    if not os.path.exists(path):
        pytest.skip(f"{path} not trained yet")
    from tensorflow.keras.models import load_model
    m = load_model(path, compile=False)
    assert m.output_shape[-1] == n, f"{path} outputs {m.output_shape[-1]}, card says {n}"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_model_contracts.py -v`
Expected: passes for every trained model.

- [ ] **Step 3: Update `app.py` model loading**

Replace `get_brain_tumor_model` and `get_satellite_model` with registry-backed
loaders that call `load_card` then `validate_card` and return `None` on
`RegistryError`, logging the reason. Point `SATELLITE_MODEL_FILE` at
`model/satellite_best.keras`, set `SATELLITE_CLASSES` to the 10 EuroSAT classes
with a `SATELLITE_GROUPS` mapping to the existing 4 display groups, and add
`ECG_MODEL_FILE = "model/ecg_cnn.keras"` with AAMI class names.

- [ ] **Step 4: Run the full application suite**

Run: `.venv/bin/python /tmp/.../verify.py`
Expected: 0 failures.

- [ ] **Step 5: Commit**

```bash
git add app.py templates tests/test_model_contracts.py && git commit -m "feat: load models through the registry and adopt EuroSAT/AAMI taxonomies"
```

---

### Task 10: Dataset documentation

**Files:**
- Create: `data/DATASET.md`
- Test: `tests/test_dataset_docs.py`

- [ ] **Step 1: Write `data/DATASET.md`**

Document, per domain: source URL, licence, download command, class list, split
policy and rationale, per-split counts, known caveats. State explicitly that ECG
uses an inter-patient split and why its metrics are lower than beat-split
publications.

- [ ] **Step 2: Write the test**

```python
import os, pytest
pytestmark = pytest.mark.skipif(not os.path.exists("data/DATASET.md"), reason="not written yet")

def test_documents_every_domain():
    text = open("data/DATASET.md").read()
    for token in ["EuroSAT", "MIT-BIH", "inter-patient", "licence", "SHA-256"]:
        assert token.lower() in text.lower(), f"DATASET.md missing {token}"
```

- [ ] **Step 3: Commit**

```bash
git add data/DATASET.md tests/test_dataset_docs.py && git commit -m "docs: add dataset card covering provenance, splits and caveats"
```

---

## Self-Review

**Spec coverage:** data architecture (Tasks 2,3,6,7,10), MRI CNN (4), MRI VGG16 (5),
EuroSAT 10-class (6), ECG 5-class AAMI inter-patient (7,8), model registry (1,9),
testing gates (1,2,3,6,7,9,10), app integration (9). No gaps.

**Placeholders:** none — every code step contains runnable code.

**Type consistency:** `save_model_card`/`load_card`/`validate_card` signatures match
across Tasks 1, 4, 5, 6, 8, 9. `sha256`/`write_manifest`/`read_manifest`/
`assert_no_split_overlap` match across Tasks 2, 3, 6, 7. Class lists are duplicated
verbatim where used rather than imported, so a task read in isolation is complete.
