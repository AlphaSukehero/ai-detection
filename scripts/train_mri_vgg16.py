"""VGG16 transfer learning for brain MRI, with cached bottleneck features."""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from mlkit.registry import save_model_card

SEED, SIZE, BATCH = 42, 224, 32
EPOCHS = int(os.environ.get("EPOCHS", "60"))
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CACHE = "data/processed/mri_vgg16_features"
tf.keras.utils.set_random_seed(SEED)


def raw_loader(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"data/processed/mri/{split}", labels="inferred", label_mode="categorical",
        class_names=CLASSES, image_size=(SIZE, SIZE), batch_size=BATCH, shuffle=False)
    return ds.map(lambda x, y: (preprocess_input(x), y),
                  num_parallel_calls=tf.data.AUTOTUNE)


def features(split, base):
    os.makedirs(CACHE, exist_ok=True)
    fx, fy = f"{CACHE}/{split}_x.npy", f"{CACHE}/{split}_y.npy"
    if os.path.exists(fx):
        return np.load(fx), np.load(fy)
    xs, ys = [], []
    for i, (bx, by) in enumerate(raw_loader(split)):
        xs.append(base.predict(bx, verbose=0))
        ys.append(by.numpy())
        if i % 20 == 0:
            print(f"  {split} batch {i}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.save(fx, x)
    np.save(fy, y)
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
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation="softmax")])
    head.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy",
                 metrics=["accuracy"])
    head.fit(xtr, ytr, validation_data=(xva, yva), epochs=EPOCHS, batch_size=64,
             verbose=2, callbacks=[
                 tf.keras.callbacks.EarlyStopping(
                     "val_loss", patience=8, restore_best_weights=True)])

    y_true, y_pred = np.argmax(yte, 1), np.argmax(head.predict(xte, verbose=0), 1)
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("confusion matrix:\n", confusion_matrix(y_true, y_pred))

    full = models.Sequential([layers.Input((SIZE, SIZE, 3)), base, head])
    os.makedirs("model", exist_ok=True)
    path = "model/mri_vgg16.keras"
    full.save(path)
    save_model_card(path, "mri", CLASSES, [SIZE, SIZE, 3], "vgg16_preprocess_input",
                    {"test_accuracy": report["accuracy"],
                     "macro_f1": report["macro avg"]["f1-score"]})
    print(f"saved {path} accuracy={report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
