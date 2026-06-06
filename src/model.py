"""
Neural-network model definition.

A small Keras ``Sequential`` classifier that maps the 10-dimensional feature
vector to one of four clustering algorithms (KMeans, DBSCAN, OPTICS, GMM):

    Dense(48, ReLU, L2) -> Dropout(0.5) -> BatchNorm
    Dense(32, ReLU, L2) -> Dropout(0.5) -> BatchNorm
    Dense(4, softmax)

Optimiser Adam (lr=1e-4), loss sparse categorical cross-entropy. The
hyperparameters match the original seminar-project notebook.
"""

from __future__ import annotations

# Hyperparameters (identical to the notebook).
NEURONS_LAYER1 = 48
NEURONS_LAYER2 = 32
L2_REG = 0.01
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Class index -> algorithm name (matches the label encoding used in training).
CLASS_NAMES = {0: "KMeans", 1: "DBSCAN", 2: "OPTICS", 3: "GMM"}


def build_model(input_dim: int = 10):
    """Build and compile the classifier.

    Imports TensorFlow lazily so the rest of the package can be used (e.g. the
    feature pipeline) without TensorFlow installed.
    """
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Dense(
                NEURONS_LAYER1,
                activation="relu",
                input_dim=input_dim,
                kernel_regularizer=regularizers.l2(L2_REG),
            ),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            Dense(
                NEURONS_LAYER2,
                activation="relu",
                kernel_regularizer=regularizers.l2(L2_REG),
            ),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            Dense(4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
