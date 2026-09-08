"""1D CNN over 280-sample MIT-BIH beats, 5 AAMI classes.

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
from mlkit.registry import save_model_card

SEED = 42
CLASSES = ["N", "S", "V", "F", "Q"]
tf.keras.utils.set_random_seed(SEED)


def load(split):
    data = np.load(f"data/processed/ecg/{split}.npz")
    return data["X"][..., None].astype("float32"), data["y"]


def build():
    model = models.Sequential([layers.Input((280, 1))])
    for filters, kernel in [(32, 7), (64, 5), (128, 3)]:
        model.add(layers.Conv1D(filters, kernel, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling1D(2))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    x_train, y_train = load("train")
    x_val, y_val = load("val")
    x_test, y_test = load("test")
    print(f"train {x_train.shape} val {x_val.shape} test {x_test.shape}")

    present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}
    print("class weights:", {CLASSES[k]: round(v, 2) for k, v in class_weight.items()})

    model = build()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=40,
              batch_size=128, class_weight=class_weight, verbose=2, callbacks=[
                  tf.keras.callbacks.EarlyStopping("val_loss", patience=6,
                                                   restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])

    y_pred = np.argmax(model.predict(x_test, verbose=0), 1)
    labels = list(range(len(CLASSES)))
    report = classification_report(y_test, y_pred, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=CLASSES, zero_division=0))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=labels))

    path = "model/ecg_cnn.keras"
    model.save(path)
    save_model_card(path, "ecg", CLASSES, [280, 1], "beat_zscore",
                    {"test_accuracy": report["accuracy"],
                     "macro_f1": report["macro avg"]["f1-score"],
                     "per_class_f1": {c: report[c]["f1-score"] for c in CLASSES}})
    print(f"saved {path} macro_F1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
