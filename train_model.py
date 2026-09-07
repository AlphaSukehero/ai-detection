import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

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
# 2. IMAGE SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32

NUM_CLASSES = 4


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


# ============================================================
# 5. LOAD TESTING DATA
# ============================================================

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
# 7. CREATE CNN MODEL
# ============================================================

model = Sequential([

    # First convolution block
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D((2, 2)),


    # Second convolution block
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),


    # Third convolution block
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),


    # Convert feature maps to vector
    Flatten(),


    # Fully connected layer
    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),


    # Four output classes
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 8. DISPLAY MODEL STRUCTURE
# ============================================================

model.summary()


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

EPOCHS = 15

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "brain_tumor_cnn.keras"
)

model.save(model_path)

print("\nModel saved successfully!")
print("Location:", model_path)


# ============================================================
# 12. TEST MODEL
# ============================================================

test_generator.reset()

predictions = model.predict(test_generator)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes

class_names = list(
    test_generator.class_indices.keys()
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ============================================================
# 14. CONFUSION MATRIX
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

plt.title("Brain Tumor Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# ============================================================
# 15. ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Testing Accuracy"
)

plt.title("Training and Testing Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    "accuracy_graph.png"
)

plt.show()


# ============================================================
# 16. LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Testing Loss"
)

plt.title("Training and Testing Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    "loss_graph.png"
)

plt.show()