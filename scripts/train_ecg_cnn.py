"""1D CNN over 280-sample MIT-BIH beats plus RR context, 5 AAMI classes.

Two changes from the morphology-only version, both aimed at the same
failure: S recall near zero on an inter-patient split.

1. RR-interval context as a second input. A supraventricular ectopic beat is
   defined by prematurity, and the beat window is z-scored and centred on the
   R peak, which normalises that away. Measured on the training split, mean
   pre-RR is 1.03x the record median for N and 0.66x for S -- a separation
   the morphology branch structurally cannot see.

2. The morphology branch ends in Flatten, not GlobalAveragePooling1D. Global
   average pooling is translation-invariant: it discards where in the window
   a deflection occurred and keeps only that it occurred. Measured against
   the previous checkpoint, shifting every test beat by 140 samples moved
   macro-F1 from 0.285 to 0.279 -- the model was very nearly blind to
   temporal structure, which is most of what separates these classes.

Metrics are reported per class because N is roughly 90% of beats: overall
accuracy would look excellent while S recall sat near zero.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ecg.beats import N_RR_FEATURES
from mlkit.registry import save_model_card

SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "40"))
CLASSES = ["N", "S", "V", "F", "Q"]
# Classes with too few training examples to learn at all. Their metrics are
# still reported in full; this list only controls the headline macro-F1, so
# that the number is not dominated by classes the dataset cannot teach.
# DS1/DS2 excludes paced records by design, which is why Q is nearly empty.
MIN_SUPPORT = 100
# Ceiling on the balanced class weight. "Balanced" weighting makes every
# class contribute equal total loss, which stops being balance when a class
# has six examples: Q's 6 beats would carry the same gradient as N's 36913,
# letting a handful of possibly-mislabelled beats steer the whole model. A
# first attempt without this cap reached val_accuracy 0.26 after one epoch,
# with the model chasing F and Q at the expense of everything else. The cap
# keeps their signal without letting it dominate.
MAX_CLASS_WEIGHT = 20.0
tf.keras.utils.set_random_seed(SEED)


def load(split):
    data = np.load(f"data/processed/ecg/{split}.npz")
    return (data["X"][..., None].astype("float32"),
            data["R"].astype("float32"),
            data["y"])


def build():
    beat_in = layers.Input((280, 1), name="beat")
    x = beat_in
    for filters, kernel in [(32, 7), (64, 5), (128, 3)]:
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    # Flatten rather than GlobalAveragePooling1D: position within the beat is
    # the signal, not a nuisance to be averaged out.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)

    rr_in = layers.Input((N_RR_FEATURES,), name="rr")
    r = layers.Dense(32, activation="relu")(rr_in)
    r = layers.Dense(32, activation="relu")(r)

    merged = layers.Concatenate()([x, r])
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(len(CLASSES), activation="softmax")(merged)

    model = models.Model([beat_in, rr_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    x_train, r_train, y_train = load("train")
    x_val, r_val, y_val = load("val")
    x_test, r_test, y_test = load("test")
    print(f"train {x_train.shape} val {x_val.shape} test {x_test.shape}")

    present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present, y=y_train)
    capped = np.minimum(weights, MAX_CLASS_WEIGHT)
    class_weight = {int(c): float(w) for c, w in zip(present, capped, strict=True)}
    raw = {CLASSES[int(c)]: round(float(w), 1)
           for c, w in zip(present, weights, strict=True)}
    print("class weights (balanced):", raw)
    print(f"class weights (capped at {MAX_CLASS_WEIGHT}):",
          {CLASSES[k]: round(v, 2) for k, v in class_weight.items()})

    model = build()
    model.fit({"beat": x_train, "rr": r_train}, y_train,
              validation_data=({"beat": x_val, "rr": r_val}, y_val),
              epochs=EPOCHS, batch_size=128, class_weight=class_weight,
              verbose=2, callbacks=[
                  tf.keras.callbacks.EarlyStopping("val_loss", patience=6,
                                                   restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])

    y_pred = np.argmax(model.predict({"beat": x_test, "rr": r_test}, verbose=0), 1)
    labels = list(range(len(CLASSES)))
    report = classification_report(y_test, y_pred, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=CLASSES, zero_division=0))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=labels))

    per_class_f1 = {c: report[c]["f1-score"] for c in CLASSES}
    supported = [c for c in CLASSES
                 if int((y_train == CLASSES.index(c)).sum()) >= MIN_SUPPORT]
    macro_supported = float(np.mean([per_class_f1[c] for c in supported]))
    print(f"macro_F1(all)={report['macro avg']['f1-score']:.4f}  "
          f"macro_F1({'+'.join(supported)})={macro_supported:.4f}")

    path = "model/ecg_cnn.keras"
    model.save(path)
    save_model_card(path, "ecg", CLASSES, [280, 1], "beat_zscore",
                    {"test_accuracy": report["accuracy"],
                     "max_class_weight": MAX_CLASS_WEIGHT,
                     "macro_f1": report["macro avg"]["f1-score"],
                     "macro_f1_supported": macro_supported,
                     "supported_classes": supported,
                     "per_class_f1": per_class_f1},
                    extra={"aux_inputs": {"rr": N_RR_FEATURES},
                           "train_support": {c: int((y_train == i).sum())
                                             for i, c in enumerate(CLASSES)}})
    print(f"saved {path} macro_F1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
