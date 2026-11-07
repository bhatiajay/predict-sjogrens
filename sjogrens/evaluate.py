"""Cross-validation: estimate how well each model does on samples it hasn't seen.

Every model is scored on exactly the same train/test splits, so the comparison is
fair:
  - neural_net           the Keras network from sjogrens.model
  - logistic_regression  a simple linear model; if the network can't beat it, the
                         extra complexity isn't paying off
  - majority_baseline    always guesses the most common class ("patient"). It gets
                         ~82% accuracy while knowing nothing, which is why accuracy
                         alone is misleading here.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sjogrens  # noqa: F401
import keras

from sjogrens import config
from sjogrens.model import predict_scores, train

MODEL_NAMES = ["neural_net", "logistic_regression", "majority_baseline"]
METRICS = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "accuracy"]


def score_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Metrics for one test fold.

    roc_auc            chance a random patient scores higher than a random control
                       (0.5 = coin flip, 1.0 = perfect). Doesn't depend on a threshold.
    sensitivity        share of patients flagged as patients
    specificity        share of controls flagged as controls
    balanced_accuracy  average of sensitivity and specificity
    """
    predicted = scores >= threshold
    patients, controls = y_true == 1, y_true == 0
    sensitivity = predicted[patients].mean()
    specificity = (~predicted[controls]).mean()
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (predicted == patients).mean(),
    }


def fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = config.SEED,
) -> dict[str, np.ndarray]:
    """Fit every model on the training split; return each one's scores for the test split."""
    keras.backend.clear_session()  # free the previous fold's model
    net = train(X_train, y_train, epochs=epochs, batch_size=batch_size, seed=seed)
    logreg = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return {
        "neural_net": predict_scores(net, X_test),
        "logistic_regression": logreg.predict_proba(X_test)[:, 1],
        "majority_baseline": baseline.predict_proba(X_test)[:, 1],
    }


def out_of_fold_scores(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, epochs: int = 100, seed: int = config.SEED
) -> dict[str, np.ndarray]:
    """One pass of stratified k-fold. Every sample gets a score from a model that never
    saw it during training. Useful for plotting ROC curves."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_scores = fit_and_score(X[train_idx], y[train_idx], X[test_idx], epochs=epochs, seed=seed + fold)
        for name, s in fold_scores.items():
            scores[name][test_idx] = s
    return scores


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = config.SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run repeated stratified k-fold CV and return one row per (fold, model).

    "Stratified" keeps the 82/18 patient/control ratio in every fold, so no test fold
    ends up with zero controls. "Repeated" reshuffles and runs it again, because with
    only 29 controls a single split can be lucky or unlucky.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    total = n_splits * n_repeats

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_scores = fit_and_score(
            X[train_idx], y[train_idx], X[test_idx], epochs=epochs, batch_size=batch_size, seed=seed + fold
        )
        for name, scores in fold_scores.items():
            rows.append({"fold": fold, "model": name, **score_metrics(y[test_idx], scores)})

        if verbose:
            nn_auc = rows[-3]["roc_auc"]
            print(f"  fold {fold + 1:2d}/{total}  neural_net AUC = {nn_auc:.3f}", flush=True)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds, one row per model."""
    grouped = results.groupby("model")[METRICS]
    summary = grouped.mean().round(3).add_suffix("_mean").join(grouped.std().round(3).add_suffix("_std"))
    ordered = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    return summary[ordered].reindex(MODEL_NAMES)


def format_summary(summary: pd.DataFrame) -> str:
    """Readable 'mean ± std' table."""
    table = pd.DataFrame(
        {m: summary[f"{m}_mean"].map("{:.3f}".format) + " ± " + summary[f"{m}_std"].map("{:.3f}".format) for m in METRICS}
    )
    return table.to_string()
