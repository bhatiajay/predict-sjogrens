"""Load the GSE66795 subset and turn it into arrays the models can use."""

from pathlib import Path

import numpy as np
import pandas as pd

from sjogrens import config

ID_COLUMN = "GEO Accession #"
LABEL_COLUMN = "Patient Group"
GENDER_COLUMN = "Gender"

# The CSV stores GEO's raw text labels. 1 = primary Sjögren's patient, 0 = healthy control.
LABELS = {"patient group: Patient": 1, "patient group: Control": 0}


def load_frame(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV, check it, and return one row per sample with gene-symbol columns.

    The result has columns: sample_id, label (0/1), then one column per gene in
    config.ALL_GENES. Gender is dropped because every sample in this dataset is
    female, so it can't help tell patients from controls.

    Raises ValueError if columns are missing, a label is unexpected, or an
    expression value is missing or non-numeric. Failing loudly here is what stops a
    silent mistake like the old `'Sjogren' in x` check, which labeled every sample 0.
    """
    raw = pd.read_csv(path)

    expected = [ID_COLUMN, LABEL_COLUMN, GENDER_COLUMN, *config.PROBE_TO_GENE]
    missing = [col for col in expected if col not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    unknown = sorted(set(raw[LABEL_COLUMN]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected values in '{LABEL_COLUMN}': {unknown}")

    expression = raw[list(config.PROBE_TO_GENE)].apply(pd.to_numeric, errors="coerce")
    if expression.isna().any().any():
        bad = expression.columns[expression.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric expression values in: {bad}")

    frame = expression.rename(columns=config.PROBE_TO_GENE)
    frame.insert(0, "label", raw[LABEL_COLUMN].map(LABELS).astype("int64"))
    frame.insert(0, "sample_id", raw[ID_COLUMN])
    return frame


def load_dataset(
    path: Path | str = config.DATA_PATH, genes: list[str] = config.REPORT_GENES
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X is float32 with one column per gene (in `genes` order), y is 0/1."""
    unknown = [g for g in genes if g not in config.GENE_TO_PROBE]
    if unknown:
        raise ValueError(f"Unknown genes: {unknown}. Choose from {config.ALL_GENES}")

    frame = load_frame(path)
    X = frame[genes].to_numpy(dtype="float32")
    y = frame["label"].to_numpy(dtype="int64")
    return X, y
