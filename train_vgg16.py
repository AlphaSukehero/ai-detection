# ============================================================
# VGG16 FOR BRAIN TUMOR MRI CLASSIFICATION
# Classes:
# 0 - Glioma
# 1 - Meningioma
# 2 - No Tumor
# 3 - Pituitary
# ============================================================


import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# 3. CHECK DATASET PATHS
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing folder not found: {TEST_DIR}"
    )


print("\nDataset paths found successfully.")


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================
#
# VGG16 was trained using ImageNet preprocessing.
# Therefore, we use preprocess_input instead of simply
# dividing pixel values by 255.
#
# 20% of the Training folder is used for validation.
# The Testing folder remains untouched until final testing.
# ============================================================


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    validation_split=0.20,

    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,

    horizontal_flip=True
)


# ============================================================
# 5. TRAINING DATA
# ============================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=42
)


# ============================================================
# 6. VALIDATION DATA
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
# 7. TEST DATA
# ============================================================
#
# IMPORTANT:
# The Testing folder is NOT used during training.
# It is used only for final evaluation.
# ============================================================


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# 8. DISPLAY CLASS LABELS
# ============================================================

print("\n========================================")
print("CLASS LABELS")
print("========================================")

for class_name, class_number in train_generator.class_indices.items():

    print(
        f"{class_number} = {class_name}"
    )


# ============================================================
# 9. LOAD PRETRAINED VGG16
# ============================================================

print("\nLoading VGG16 model...")

base_model = VGG16(

    include_top=False,

    weights="imagenet",

    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# ============================================================
# 10. FREEZE VGG16 LAYERS
# ============================================================
#
# Initially, the pretrained VGG16 layers are frozen.
# Only our new classification layers will learn.
# ============================================================


base_model.trainable = False


# ============================================================
# 11. CREATE NEW CLASSIFICATION HEAD
# ============================================================


inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)


# Pass image through VGG16

x = base_model(
    inputs,
    training=False
)


# Reduce feature maps

x = layers.GlobalAveragePooling2D()(x)


# Fully connected layer

x = layers.Dense(
    256,
    activation="relu"
)(x)


# Dropout to reduce overfitting

x = layers.Dropout(
    0.5
)(x)


# Final four-class output

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


# Create complete model

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# 12. DISPLAY MODEL
# ============================================================

print("\n========================================")
print("VGG16 MODEL")
print("========================================")

model.summary()


# ============================================================
# 13. COMPILE MODEL
# ============================================================


model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 14. CALLBACKS
# ============================================================


checkpoint = ModelCheckpoint(

    filepath="model/vgg16_best.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# 15. TRAIN VGG16
# ============================================================

print("\n========================================")
print("STARTING VGG16 TRAINING")
print("========================================")


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
# 16. FINAL TEST EVALUATION
# ============================================================

print("\n========================================")
print("FINAL TESTING")
print("========================================")


test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\n========================================")
print("FINAL VGG16 TEST RESULT")
print("========================================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# 17. GENERATE PREDICTIONS
# ============================================================


print("\nGenerating predictions...")

test_generator.reset()


predictions = model.predict(
    test_generator,
    verbose=1
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
# 18. CLASSIFICATION REPORT
# ============================================================


print("\n========================================")
print("VGG16 CLASSIFICATION REPORT")
print("========================================")


report = classification_report(

    true_classes,

    predicted_classes,

    target_names=class_names,

    digits=4
)


print(report)


# Save report to a text file

with open(
    "vgg16_classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# 19. CALCULATE OVERALL METRICS
# ============================================================


accuracy = accuracy_score(
    true_classes,
    predicted_classes
)


precision = precision_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)


print("\n========================================")
print("OVERALL METRICS")
print("========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================


cm = confusion_matrix(

    true_classes,

    predicted_classes
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


disp.plot(
    xticks_rotation=45
)


plt.title(
    "VGG16 Brain Tumor Classification"
)

plt.tight_layout()


plt.savefig(
    "vgg16_confusion_matrix.png",
    dpi=300
)


plt.show()


# ============================================================
# 21. TRAINING ACCURACY GRAPH
# ============================================================


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)


plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)


plt.title(
    "VGG16 Training and Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(
    "vgg16_accuracy.png",
    dpi=300
)


plt.show()


# ============================================================
# 22. TRAINING LOSS GRAPH
# ============================================================


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    history.history["loss"],

    label="Training Loss"
)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)


plt.title(
    "VGG16 Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(
    "vgg16_loss.png",
    dpi=300
)


plt.show()


# ============================================================
# 23. SAVE FINAL MODEL
# ============================================================


final_model_path = (
    "model/vgg16_brain_tumor.keras"
)


model.save(
    final_model_path
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    f"Saved at: {final_model_path}"
)


print("\nVGG16 training and testing completed successfully!")