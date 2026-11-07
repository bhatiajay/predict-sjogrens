import numpy as np
import pytest

from sjogrens import artifacts
from sjogrens.data import load_dataset
from sjogrens.model import class_weights, predict_scores, train


def test_class_weights_balance_the_classes():
    y = np.array([1] * 131 + [0] * 29)
    w = class_weights(y)
    assert w[0] == pytest.approx(160 / 58)
    assert w[1] == pytest.approx(160 / 262)
    # Each class now contributes the same total weight.
    assert w[0] * 29 == pytest.approx(w[1] * 131)


def test_save_load_roundtrip_gives_identical_scores(tmp_path):
    X, y = load_dataset()
    model = train(X, y, epochs=5)
    meta = artifacts.build_meta(X, y, ["IRF5", "SSB", "TRIM21", "MX1", "IFI44"], cv_summary=None)

    artifacts.save(model, meta, tmp_path)
    loaded = artifacts.load(tmp_path)
    assert loaded is not None
    loaded_model, loaded_meta = loaded

    # The standardization stats travel inside the .keras file, so raw inputs
    # give the same output before and after reloading.
    np.testing.assert_allclose(predict_scores(model, X), predict_scores(loaded_model, X), rtol=1e-5)
    assert loaded_meta["genes"] == meta["genes"]
    assert loaded_meta["n_patients"] == 131 and loaded_meta["n_controls"] == 29


def test_scores_are_probabilities():
    X, y = load_dataset()
    scores = predict_scores(train(X, y, epochs=5), X)
    assert scores.shape == (160,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_load_returns_none_when_not_trained(tmp_path):
    assert artifacts.load(tmp_path) is None
