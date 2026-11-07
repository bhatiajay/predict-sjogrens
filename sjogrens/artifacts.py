"""Save and load what the web app needs: the trained model plus a metadata file."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

import sjogrens  # noqa: F401
import keras

from sjogrens import config


def build_meta(X: np.ndarray, y: np.ndarray, genes: list[str], cv_summary: dict | None) -> dict:
    """Describe the training data so the app can label inputs and sanity-check them."""
    patients, controls = X[y == 1], X[y == 0]
    return {
        "genes": genes,
        "probes": {g: config.GENE_TO_PROBE[g] for g in genes},
        "dataset": "GSE66795 (Illumina HumanHT-12 v4, whole blood, log2 intensities)",
        "n_samples": int(len(y)),
        "n_patients": int(len(patients)),
        "n_controls": int(len(controls)),
        # Values outside the training range get a warning in the app.
        "ranges": {g: [float(X[:, i].min()), float(X[:, i].max())] for i, g in enumerate(genes)},
        # Median profile of each group, used for the "fill example values" links.
        "examples": {
            "patient": {g: round(float(np.median(patients[:, i])), 3) for i, g in enumerate(genes)},
            "control": {g: round(float(np.median(controls[:, i])), 3) for i, g in enumerate(genes)},
        },
        "cross_validation": cv_summary,
        "versions": {pkg: version(pkg) for pkg in ("tensorflow", "keras", "scikit-learn", "numpy")},
    }


def save(model: keras.Model, meta: dict, out_dir: Path | str = config.MODEL_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / config.MODEL_FILE)
    (out_dir / config.META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def load(model_dir: Path | str = config.MODEL_DIR) -> tuple[keras.Model, dict] | None:
    """Return (model, meta), or None if the model hasn't been trained yet."""
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / config.MODEL_FILE, model_dir / config.META_FILE
    if not (model_path.exists() and meta_path.exists()):
        return None
    return keras.models.load_model(model_path), json.loads(meta_path.read_text())
