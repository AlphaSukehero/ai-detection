# ============================================================
# HYBRID CNN + QUANTUM CNN (QCNN)
# Brain Tumor MRI Classification
#
# Classes:
# 0 = glioma
# 1 = meningioma
# 2 = notumor
# 3 = pituitary
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import pennylane as qml

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_CLASSES = 4

NUM_QUBITS = 4

EPOCHS_CLASSICAL = 5

EPOCHS_QUANTUM = 15

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing folder not found: {TEST_DIR}"
    )


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform_train = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


transform_test = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform_train
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nClass labels:")

for name, index in train_dataset.class_to_idx.items():

    print(index, "=", name)


# ============================================================
# 5. LOAD PRETRAINED CNN
# ============================================================

print("\nLoading EfficientNetB0...")


weights = models.EfficientNet_B0_Weights.DEFAULT

cnn = models.efficientnet_b0(
    weights=weights
)


# Remove original classifier

cnn.classifier = nn.Identity()


# Freeze CNN initially

for parameter in cnn.parameters():

    parameter.requires_grad = False


cnn = cnn.to(DEVICE)


# ============================================================
# 6. FEATURE EXTRACTION
# ============================================================

def extract_features(
    model,
    loader
):

    features = []

    labels = []

    model.eval()

    with torch.no_grad():

        for images, target in loader:

            images = images.to(DEVICE)

            output = model(images)

            features.append(
                output.cpu()
            )

            labels.append(
                target
            )

    features = torch.cat(
        features
    )

    labels = torch.cat(
        labels
    )

    return features, labels


print("\nExtracting CNN features...")

train_features, train_labels = extract_features(
    cnn,
    train_loader
)


test_features, test_labels = extract_features(
    cnn,
    test_loader
)


print(
    "Original feature size:",
    train_features.shape
)


# ============================================================
# 7. FEATURE REDUCTION
# ============================================================

# EfficientNet produces 1280 features.
# We reduce them to 4 quantum features.

from sklearn.decomposition import PCA

pca = PCA(
    n_components=NUM_QUBITS
)


train_reduced = pca.fit_transform(
    train_features.numpy()
)


test_reduced = pca.transform(
    test_features.numpy()
)


# Normalize quantum features to [0, 1]

minimum = train_reduced.min(
    axis=0
)

maximum = train_reduced.max(
    axis=0
)


train_reduced = (
    train_reduced - minimum
) / (
    maximum - minimum + 1e-8
)


test_reduced = (
    test_reduced - minimum
) / (
    maximum - minimum + 1e-8
)


# Convert to tensors

train_reduced = torch.tensor(
    train_reduced,
    dtype=torch.float32
)


test_reduced = torch.tensor(
    test_reduced,
    dtype=torch.float32
)


train_labels = train_labels.long()

test_labels = test_labels.long()


print(
    "Reduced feature size:",
    train_reduced.shape
)


# ============================================================
# 8. QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=NUM_QUBITS
)


# ============================================================
# 9. QUANTUM CONVOLUTION
# ============================================================

def quantum_convolution():

    # Local two-qubit interactions

    for i in range(
        NUM_QUBITS - 1
    ):

        qml.CNOT(
            wires=[i, i + 1]
        )

        qml.RY(
            0.5,
            wires=i
        )

        qml.RZ(
            0.5,
            wires=i + 1
        )


# ============================================================
# 10. QUANTUM CIRCUIT
# ============================================================

@qml.qnode(
    dev,
    interface="torch"
)
def quantum_circuit(
    inputs,
    weights
):

    # --------------------------------------------------------
    # Quantum feature encoding
    # --------------------------------------------------------

    for i in range(
        NUM_QUBITS
    ):

        qml.RY(
            np.pi * inputs[i],
            wires=i
        )

        qml.RZ(
            np.pi * inputs[i],
            wires=i
        )


    # --------------------------------------------------------
    # Quantum convolution layers
    # --------------------------------------------------------

    for layer in range(
        2
    ):

        for i in range(
            NUM_QUBITS
        ):

            qml.RY(
                weights[layer, i, 0],
                wires=i
            )

            qml.RZ(
                weights[layer, i, 1],
                wires=i
            )


        quantum_convolution()


    # --------------------------------------------------------
    # Quantum pooling-style reduction
    # --------------------------------------------------------

    qml.CNOT(
        wires=[0, 1]
    )

    qml.CNOT(
        wires=[2, 3]
    )


    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(
            NUM_QUBITS
        )
    ]


# ============================================================
# 11. QUANTUM CLASSIFIER
# ============================================================

class QuantumClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.q_weights = nn.Parameter(
            0.01 * torch.randn(
                2,
                NUM_QUBITS,
                2,
                dtype=torch.float32
            )
        )

        self.classifier = nn.Linear(
            NUM_QUBITS,
            NUM_CLASSES
        ).float()


    def forward(self, x):

        quantum_outputs = []

        for sample in x:

            sample = sample.float()

            q_result = quantum_circuit(
                sample,
                self.q_weights
            )

            q_result = torch.stack(
                q_result
            ).float()

            quantum_outputs.append(
                q_result
            )

        quantum_outputs = torch.stack(
            quantum_outputs
        ).float()

        output = self.classifier(
            quantum_outputs
        )

        return output
# ============================================================
# 12. CREATE MODEL
# ============================================================

q_model = QuantumClassifier()

# ============================================================
# 12. CREATE MODEL
# ============================================================

q_model = QuantumClassifier()

q_model = q_model.to(DEVICE)

print("\nQuantum model:")


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    q_model.parameters(),
    lr=LEARNING_RATE
)


criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. CREATE QUANTUM DATA LOADERS
# ============================================================

quantum_train_dataset = torch.utils.data.TensorDataset(
    train_reduced,
    train_labels
)


quantum_test_dataset = torch.utils.data.TensorDataset(
    test_reduced,
    test_labels
)


quantum_train_loader = DataLoader(
    quantum_train_dataset,
    batch_size=8,
    shuffle=True
)


quantum_test_loader = DataLoader(
    quantum_test_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================================
# 15. TRAIN QCNN
# ============================================================

print("\n========================================")
print("TRAINING QUANTUM CNN")
print("========================================")


for epoch in range(
    EPOCHS_QUANTUM
):

    q_model.train()

    total_loss = 0

    correct = 0

    total = 0


    for features, labels in quantum_train_loader:

        features = features.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = q_model(
            features
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    accuracy = (
        correct / total
    ) * 100


    print(
        f"Epoch {epoch + 1}/{EPOCHS_QUANTUM} "
        f"- Loss: {total_loss:.4f} "
        f"- Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# 16. FINAL TEST
# ============================================================

print("\n========================================")
print("FINAL QUANTUM TEST")
print("========================================")


q_model.eval()


all_predictions = []

all_labels = []


with torch.no_grad():

    for features, labels in quantum_test_loader:

        features = features.to(
            DEVICE
        )


        outputs = q_model(
            features
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# 17. RESULTS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print(
    f"\nQCNN Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "glioma",
            "meningioma",
            "notumor",
            "pituitary"
        ],
        digits=4
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 19. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


torch.save(
    q_model.state_dict(),
    "model/qcnn_brain_tumor.pth"
)


print(
    "\nQCNN model saved successfully!"
)


# ============================================================
# 20. SAVE PCA
# ============================================================

import joblib

joblib.dump(
    pca,
    "model/qcnn_pca.pkl"
)


print(
    "PCA model saved successfully!"
)