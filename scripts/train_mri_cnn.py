"""Scratch CNN for 4-class brain MRI classification at 128x128."""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
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
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build():
    model = models.Sequential([layers.Input((SIZE, SIZE, 3))])
    for filters in [32, 64, 128, 128]:
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train, val, test = loader("train", True, True), loader("val", False), loader("test", False)
    model = build()
    model.fit(train, validation_data=val, epochs=EPOCHS, verbose=2, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3)])

    y_true = np.concatenate([np.argmax(y, 1) for _, y in test])
    y_pred = np.argmax(model.predict(test, verbose=0), 1)
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("confusion matrix:\n", confusion_matrix(y_true, y_pred))

    os.makedirs("model", exist_ok=True)
    path = "model/mri_cnn.keras"
    model.save(path)
    save_model_card(path, "mri", CLASSES, [SIZE, SIZE, 3], "rescale_255",
                    {"test_accuracy": report["accuracy"],
                     "macro_f1": report["macro avg"]["f1-score"]})
    print(f"saved {path} accuracy={report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
