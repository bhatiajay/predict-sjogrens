"""The neural network: how it's built, trained, and used to score samples."""

import numpy as np

import sjogrens  # noqa: F401  (sets TF log level before Keras loads)
import keras
from keras import layers, regularizers

from sjogrens import config


def build_model(X_train: np.ndarray, learning_rate: float = 1e-3) -> keras.Model:
    """Create and compile the network.

    The first layer standardizes each gene (subtract mean, divide by std). It learns
    those statistics from X_train only, then saves them inside the model file. That
    keeps the scaling identical in training, evaluation and the web app, and stops
    test-fold data from leaking into the scaling.

    The rest is the architecture from the original project: two hidden layers with
    L2 weight penalties and dropout to limit overfitting on a small dataset, and one
    sigmoid output between 0 and 1.
    """
    standardize = layers.Normalization(name="standardize")
    standardize.adapt(X_train)

    model = keras.Sequential(
        [
            keras.Input(shape=(X_train.shape[1],), name="genes"),
            standardize,
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="sjogrens_net",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Loss weights that give both classes equal total influence.

    With 131 patients and 29 controls, an unweighted model can score well by
    leaning toward "patient" every time. Weighting each class by n / (2 * count)
    makes a mistake on a control cost about 4.5x more than one on a patient.
    """
    counts = np.bincount(y, minlength=2)
    return {label: float(len(y) / (2 * counts[label])) for label in (0, 1)}


def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = config.SEED,
) -> keras.Model:
    """Build a fresh model and fit it on (X, y)."""
    keras.utils.set_random_seed(seed)
    model = build_model(X)
    model.fit(
        X, y, epochs=epochs, batch_size=batch_size, class_weight=class_weights(y), verbose=0
    )
    return model


def predict_scores(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Return the model's output (0-1) for each row of X as a flat array."""
    X = np.asarray(X, dtype="float32")
    return keras.ops.convert_to_numpy(model(X, training=False)).ravel()
