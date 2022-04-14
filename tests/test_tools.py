import numpy as np
import pytest

from score_analysis import Scores, tools


@pytest.mark.parametrize("fpr, fnr", [[[0.3, 0.4], None], [None, [0.2]]])
@pytest.mark.parametrize("method", ["bootstrap", "binomial", "pessimist"])
def test_roc_curve(fpr, fnr, method):
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    res_fpr, res_fnr, lower, upper = tools.roc_with_ci(
        scores, fpr=fpr, fnr=fnr, method=method, alpha=0.1
    )
    if fpr is not None:
        np.testing.assert_equal(res_fpr, fpr)
    if fnr is not None:
        np.testing.assert_equal(res_fnr, fnr)

    assert res_fpr.shape == res_fnr.shape
    assert res_fpr.shape == lower.shape
    assert res_fnr.shape == upper.shape


def test_roc_curve_invalid_args():
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    with pytest.raises(ValueError):
        tools.roc_with_ci(scores, fnr=None, fpr=None)
    with pytest.raises(ValueError):
        tools.roc_with_ci(scores, fnr=np.array([0.3]), fpr=np.array([0.3]))
    with pytest.raises(ValueError):
        tools.roc_with_ci(scores, fnr=np.array([0.3]), method="no_such_method")
