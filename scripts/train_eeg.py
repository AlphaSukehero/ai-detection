"""Train one task head on scalogram windows.

    python scripts/train_eeg.py seizure
    python scripts/train_eeg.py alzheimer
    EPOCHS=2 python scripts/train_eeg.py seizure     # smoke test

Reported metrics are deliberately not accuracy. Seizure windows are a low
single-digit percentage of this dataset, so a model that predicts "no
seizure" everywhere scores about 97% and finds nothing. Sensitivity,
precision and AUC over the positive class are the numbers that say whether
the model works, and they go on the card.

Class weights are capped for the reason established elsewhere in this
repository: fully balanced weighting on a rare class hands a handful of
windows the same total gradient as tens of thousands, and the model chases
noise instead of learning a boundary.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from tensorflow.keras import layers, models

from eeg.masking import class_weights
from mlkit.registry import save_model_card

SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "25"))
DATA = "data/processed/eeg"
tf.keras.utils.set_random_seed(SEED)


def load(task, split):
    d = np.load(f"{DATA}/{task}/{split}.npz")
    return d["X"], d["y"], d["source"]


def shuffled(x, y, seed=SEED):
    """One fixed permutation before fit.

    Keras' validation_split takes the LAST fraction of the arrays without
    shuffling. The prepared arrays are ordered by source recording, so that
    tail can hold no positive window at all, and early stopping on val_auc
    would then be steering on noise.
    """
    order = np.random.default_rng(seed).permutation(len(y))
    return x[order], y[order]


def build(input_shape):
    """Small CNN over the time-frequency image.

    Ends in GlobalAveragePooling only over the TIME axis, keeping the
    frequency axis intact through the dense layer. Pooling both axes would
    make the model translation-invariant in frequency, which is precisely
    wrong here: delta slowing and a beta artifact differ by *where* on the
    frequency axis the energy sits, and averaging that away discards the
    distinction the clinical task depends on.
    """
    inp = layers.Input(input_shape)
    x = inp
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    # Built-in layers rather than a Lambda: Keras refuses to load a Python
    # lambda under its default safe_mode, which the dashboard uses.
    freq, time, filters = x.shape[1:]
    x = layers.AveragePooling2D((1, time), name="pool_time")(x)
    x = layers.Reshape((freq, filters))(x)          # (batch, freq, filters)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "seizure"
    x_train, y_train, src_train = load(task, "train")
    x_test, y_test, src_test = load(task, "test")
    manifest = json.load(open(f"{DATA}/{task}/manifest.json"))

    print(f"{task}: train {x_train.shape} ({int(y_train.sum())} positive) "
          f"test {x_test.shape} ({int(y_test.sum())} positive)")
    leak = set(src_train.tolist()) & set(src_test.tolist())
    if leak:
        raise SystemExit(f"sources in both splits: {sorted(leak)}")

    weights = class_weights(y_train)
    print("class weights (capped):", {k: round(v, 2) for k, v in weights.items()})

    x_train, y_train = shuffled(x_train, y_train)
    model = build(x_train.shape[1:])
    model.fit(x_train, y_train, validation_split=0.15, epochs=EPOCHS,
              batch_size=64, class_weight=weights, verbose=2, shuffle=True,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      "val_auc", mode="max", patience=5,
                      restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      "val_loss", factor=0.5, patience=3)])

    probs = model.predict(x_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True,
                                   zero_division=0,
                                   target_names=["negative", "positive"])
    print(classification_report(y_test, preds, zero_division=0,
                                target_names=["negative", "positive"]))
    print("confusion matrix:\n", confusion_matrix(y_test, preds))

    both = len(np.unique(y_test)) > 1
    auc = float(roc_auc_score(y_test, probs)) if both else None
    ap = float(average_precision_score(y_test, probs)) if both else None
    pos = report["positive"]
    print(f"sensitivity={pos['recall']:.3f} precision={pos['precision']:.3f} "
          f"AUC={auc if auc is None else round(auc, 3)} "
          f"AP={ap if ap is None else round(ap, 3)}")

    path = f"model/eeg_{task}.keras"
    model.save(path)
    save_model_card(
        path, f"eeg_{task}", ["negative", "positive"],
        list(x_train.shape[1:]), "scalogram_log_minmax",
        {"sensitivity": pos["recall"], "precision": pos["precision"],
         "f1": pos["f1-score"], "roc_auc": auc, "average_precision": ap,
         "test_positive_windows": int(y_test.sum()),
         "test_windows": int(len(y_test))},
        extra={"window_s": manifest["window_s"], "overlap": manifest["overlap"],
               "fs": manifest["fs"], "split_by": manifest["split_by"],
               "train_sources": manifest["splits"]["train"]["sources"],
               "test_sources": manifest["splits"]["test"]["sources"],
               "label_note": ("per-event annotations" if task == "seizure"
                              else "subject-level label applied to every "
                                   "window (weak labelling)")})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
