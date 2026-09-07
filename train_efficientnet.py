import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PATHS
# ============================================================

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 15


# ============================================================
# 3. TRAINING DATA
# ============================================================

train_datagen = ImageDataGenerator(
    validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# ============================================================
# 4. VALIDATION DATA
# ============================================================

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ============================================================
# 5. TEST DATA
# ============================================================

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ============================================================
# 6. DISPLAY CLASS LABELS
# ============================================================

print("\nClass labels:")

for class_name, class_number in train_generator.class_indices.items():
    print(class_number, "=", class_name)


# ============================================================
# 7. LOAD EFFICIENTNETB0
# ============================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# Freeze pretrained layers initially

base_model.trainable = False


# ============================================================
# 8. CREATE CLASSIFICATION MODEL
# ============================================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    "model/efficientnet_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


# ============================================================
# 12. TRAIN
# ============================================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# 13. FINAL TEST
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\n====================================")
print("FINAL TEST ACCURACY")
print("====================================")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# 14. PREDICTIONS
# ============================================================

test_generator.reset()

predictions = model.predict(
    test_generator
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes

class_names = list(
    test_generator.class_indices.keys()
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


disp.plot(
    xticks_rotation=45
)


plt.title(
    "EfficientNetB0 - Brain Tumor Classification"
)

plt.tight_layout()

plt.savefig(
    "efficientnet_confusion_matrix.png"
)

plt.show()


# ============================================================
# 17. ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "EfficientNetB0 Training and Validation Accuracy"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "efficientnet_accuracy.png"
)

plt.show()


# ============================================================
# 18. LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "EfficientNetB0 Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "efficientnet_loss.png"
)

plt.show()


# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

model.save(
    "model/efficientnet_brain_tumor.keras"
)

print("\nModel saved successfully!")