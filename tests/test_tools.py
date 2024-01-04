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
    roc_fnr, roc_fpr = tools.roc(scores, fnr=fnr, fpr=fpr, thresholds=thresholds)
    assert len(roc_fnr) == len(roc_fpr)
    assert len(roc_fnr) == expected_len


def test_roc_nb_points():
    scores = Scores(pos=[1, 2, 3], neg=[-1, -2, -3])
    roc_fnr, roc_fpr = tools.roc(scores, nb_points=10)
    assert len(roc_fnr) == len(roc_fpr) == 10

    roc_fnr, roc_fpr = tools.roc(scores, fnr=[], fpr=[], thresholds=[], nb_points=12)
    assert len(roc_fnr) == len(roc_fpr) == 12


@pytest.mark.parametrize("fpr, fnr", [[[0.3, 0.4], None], [None, [0.2]]])
def test_roc_with_ci(fpr, fnr):
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    res_fnr, res_fpr, fnr_ci, fpr_ci = tools.roc_with_ci(
        scores, fpr=fpr, fnr=fnr, alpha=0.1
    )

    assert res_fpr.shape == res_fnr.shape
    assert fnr_ci.shape == res_fpr.shape + (2,)
    assert fpr_ci.shape == res_fnr.shape + (2,)


def test_roc_with_ci_2():
    """Test with more data"""
    rng = np.random.default_rng(seed=42)
    scores = Scores(
        pos=rng.normal(loc=0.1, size=100), neg=rng.normal(loc=-0.1, size=100)
    )
    tools.roc_with_ci(scores, fpr=np.logspace(-2, -1))
