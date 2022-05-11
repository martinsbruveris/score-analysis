import numpy as np
import pytest

from score_analysis import Scores, tools


@pytest.mark.parametrize("fpr, fnr", [[[0.3, 0.4], None], [None, [0.2]]])
def test_roc_curve(fpr, fnr):
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    res_fpr, res_fnr, lower, upper = tools.roc_with_ci(
        scores, fpr=fpr, fnr=fnr, alpha=0.1
    )
    if fpr is not None:
        np.testing.assert_equal(res_fpr, fpr)
    if fnr is not None:
        np.testing.assert_equal(res_fnr, fnr)

    assert res_fpr.shape == res_fnr.shape
    assert res_fpr.shape == lower.shape
    assert res_fnr.shape == upper.shape


def test_roc_curve_2():
    """Test with more data"""
    rng = np.random.default_rng(seed=42)
    scores = Scores(
        pos=rng.normal(loc=0.1, size=100), neg=rng.normal(loc=-0.1, size=100)
    )
    tools.roc_with_ci(scores, fpr=np.logspace(-2, -1))


def test_roc_curve_invalid_args():
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    with pytest.raises(ValueError):
        tools.roc_with_ci(scores, fnr=None, fpr=None)
    with pytest.raises(ValueError):
        tools.roc_with_ci(scores, fnr=np.array([0.3]), fpr=np.array([0.3]))
