import numpy as np
import pandas as pd
import pytest

from sjogrens import config
from sjogrens.data import load_dataset, load_frame


def test_dataset_shape_and_labels():
    X, y = load_dataset()
    assert X.shape == (160, len(config.REPORT_GENES))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    # GSE66795: 131 primary Sjögren's patients, 29 controls.
    assert (y == 1).sum() == 131
    assert (y == 0).sum() == 29


def test_feature_columns_follow_requested_order():
    frame = load_frame()
    X, _ = load_dataset(genes=["MX1", "IRF5"])
    np.testing.assert_allclose(X[:, 0], frame["MX1"].to_numpy(dtype="float32"))
    np.testing.assert_allclose(X[:, 1], frame["IRF5"].to_numpy(dtype="float32"))


def test_all_genes_feature_set():
    X, _ = load_dataset(genes=config.ALL_GENES)
    assert X.shape == (160, 13)


def test_unknown_label_raises(tmp_path):
    raw = pd.read_csv(config.DATA_PATH)
    raw.loc[0, "Patient Group"] = "patient group: Sjogren"
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected values"):
        load_frame(path)


def test_missing_column_raises(tmp_path):
    raw = pd.read_csv(config.DATA_PATH).drop(columns=["ILMN_1662358"])
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_frame(path)


def test_non_numeric_value_raises(tmp_path):
    raw = pd.read_csv(config.DATA_PATH)
    raw["ILMN_1662358"] = raw["ILMN_1662358"].astype(object)
    raw.loc[3, "ILMN_1662358"] = "n/a"
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-numeric"):
        load_frame(path)


def test_unknown_gene_raises():
    with pytest.raises(ValueError, match="Unknown genes"):
        load_dataset(genes=["NOT_A_GENE"])
