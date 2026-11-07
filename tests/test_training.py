import numpy as np
import pytest

from sjogrens.data import load_dataset
from sjogrens.evaluate import cross_validate, score_metrics, summarize
from sjogrens.model import predict_scores, train


@pytest.fixture(scope="module")
def quick_cv():
    X, y = load_dataset()
    results = cross_validate(X, y, n_splits=3, n_repeats=1, epochs=30, verbose=False)
    return summarize(results)


def test_neural_net_learns_real_signal(quick_cv):
    # MX1 and IFI44 alone separate the groups with AUC ~0.87-0.90, so a working
    # model must be well above chance. A label or scaling bug would land near 0.5.
    assert quick_cv.loc["neural_net", "roc_auc_mean"] > 0.75


def test_logistic_regression_learns_real_signal(quick_cv):
    assert quick_cv.loc["logistic_regression", "roc_auc_mean"] > 0.75


def test_majority_baseline_shows_why_accuracy_misleads(quick_cv):
    base = quick_cv.loc["majority_baseline"]
    assert base["roc_auc_mean"] == pytest.approx(0.5)
    assert base["accuracy_mean"] > 0.8  # high accuracy...
    assert base["specificity_mean"] == 0  # ...while missing every control


def test_score_metrics_hand_checked():
    y = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.6, 0.1])
    m = score_metrics(y, scores)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["roc_auc"] == pytest.approx(5 / 6)  # 5 of 6 patient-control pairs ranked correctly


def test_typical_patient_scores_higher_than_typical_control():
    X, y = load_dataset()
    model = train(X, y, epochs=30)
    patient = np.median(X[y == 1], axis=0, keepdims=True)
    control = np.median(X[y == 0], axis=0, keepdims=True)
    assert predict_scores(model, patient)[0] > predict_scores(model, control)[0]


def test_out_of_fold_scores_cover_every_sample():
    from sklearn.metrics import roc_auc_score

    from sjogrens.evaluate import MODEL_NAMES, out_of_fold_scores

    X, y = load_dataset()
    scores = out_of_fold_scores(X, y, n_splits=3, epochs=10)
    assert list(scores) == MODEL_NAMES
    for s in scores.values():
        assert s.shape == (160,)
    assert roc_auc_score(y, scores["logistic_regression"]) > 0.75
