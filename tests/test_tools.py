import numpy as np
import pytest

from score_analysis import Scores, tools


@pytest.mark.parametrize(
    "fnr, fpr, thresholds",
    [
        ([0.1, 0.2, 0.3], None, None),
        (None, [0.3, 0.4, 0.5], None),
        (None, None, [1.0, 2.0, 3.0]),
        ([0.1, 0.2], [0.1, 0.2], [1.0, 2.0]),
    ],
)
def test_roc(fnr, fpr, thresholds):
    scores = Scores(pos=[1, 2, 3], neg=[-1, -2, -3])
    expected_len = len(fnr or []) + len(fpr or []) + len(thresholds or [])
    roc_curve = tools.roc(scores, fnr=fnr, fpr=fpr, thresholds=thresholds)
    assert len(roc_curve.fnr) == len(roc_curve.fpr)
    assert len(roc_curve.fnr) == expected_len


def test_roc_nb_points():
    scores = Scores(pos=[1, 2, 3], neg=[-1, -2, -3])
    roc_curve = tools.roc(scores, nb_points=10)
    assert len(roc_curve.fnr) == len(roc_curve.fpr) == 10

    roc_curve = tools.roc(scores, fnr=[], fpr=[], thresholds=[], nb_points=12)
    assert len(roc_curve.fnr) == len(roc_curve.fpr) == 12


@pytest.mark.parametrize("fpr, fnr", [[[0.3, 0.4], None], [None, [0.2]]])
def test_roc_with_ci(fpr, fnr):
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    roc_curve = tools.roc_with_ci(scores, fpr=fpr, fnr=fnr, alpha=0.1)

    assert roc_curve.fnr.shape == roc_curve.fpr.shape
    assert roc_curve.fnr_ci.shape == roc_curve.fnr.shape + (2,)
    assert roc_curve.fpr_ci.shape == roc_curve.fpr.shape + (2,)


def test_roc_with_ci_2():
    """Test with more data"""
    rng = np.random.default_rng(seed=42)
    scores = Scores(
        pos=rng.normal(loc=0.1, size=100), neg=rng.normal(loc=-0.1, size=100)
    )
    tools.roc_with_ci(scores, fpr=np.logspace(-2, -1))
