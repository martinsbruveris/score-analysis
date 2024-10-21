import numpy as np
import pytest

from score_analysis import ROCCurve, Scores, roc, roc_with_ci
from score_analysis.roc_curve import ROC_CI_EXTRA_POINTS


def test_roc_curve_class():
    roc_curve = ROCCurve(
        fnr=np.array([0.2, 0.3]),
        fpr=np.array([0.5, 0.6]),
        thresholds=np.array([0.0, -1.0]),
        fnr_ci=np.array([[0.1, 0.3], [0.2, 0.4]]),
        fpr_ci=np.array([[0.4, 0.6], [0.4, 0.8]]),
    )
    np.testing.assert_almost_equal(roc_curve.tpr, 1.0 - roc_curve.fnr)
    np.testing.assert_almost_equal(roc_curve.tnr, 1.0 - roc_curve.fpr)
    np.testing.assert_almost_equal(roc_curve.tpr_ci, np.array([[0.7, 0.9], [0.6, 0.8]]))
    np.testing.assert_almost_equal(roc_curve.tnr_ci, np.array([[0.4, 0.6], [0.2, 0.6]]))

    # Testing aliases
    np.testing.assert_equal(roc_curve.fnr, roc_curve.frr)
    np.testing.assert_equal(roc_curve.fpr, roc_curve.far)
    np.testing.assert_equal(roc_curve.tnr, roc_curve.trr)
    np.testing.assert_equal(roc_curve.tpr, roc_curve.tar)
    np.testing.assert_equal(roc_curve.fnr_ci, roc_curve.frr_ci)
    np.testing.assert_equal(roc_curve.fpr_ci, roc_curve.far_ci)
    np.testing.assert_equal(roc_curve.tnr_ci, roc_curve.trr_ci)
    np.testing.assert_equal(roc_curve.tpr_ci, roc_curve.tar_ci)


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
    roc_curve = roc(scores, fnr=fnr, fpr=fpr, thresholds=thresholds)
    assert len(roc_curve.fnr) == len(roc_curve.fpr)
    assert len(roc_curve.fnr) == expected_len


def test_roc_nb_points():
    scores = Scores(pos=[1, 2, 3], neg=[-1, -2, -3])
    roc_curve = roc(scores, nb_points=10)
    assert len(roc_curve.fnr) == len(roc_curve.fpr) == 10

    roc_curve = roc(scores, fnr=[], fpr=[], thresholds=[], nb_points=12)
    assert len(roc_curve.fnr) == len(roc_curve.fpr) == 12

    roc_curve = roc(scores, nb_points=None)  # Use all available scores
    assert len(roc_curve.fnr) == len(roc_curve.fpr) == scores.nb_all_samples


@pytest.mark.parametrize(
    "x_axis", ["fnr", "fpr", "tnr", "tpr", "far", "frr", "tar", "trr"]
)
@pytest.mark.parametrize("score_class", ["pos", "neg"])
def test_roc_x_axis(x_axis, score_class):
    scores = Scores(pos=[1, 2, 3], neg=[-1, -2, -3], score_class=score_class)
    roc_curve = roc(scores, nb_points=None, x_axis=x_axis)
    # Check that the x-axis metric is increasing
    assert np.all(np.diff(getattr(roc_curve, x_axis)) >= 0.0)

    with pytest.raises(ValueError):
        roc(scores, x_axis="no_such_metric")


@pytest.mark.parametrize("fpr, fnr", [[[0.3, 0.4], None], [None, [0.2]]])
def test_roc_with_ci(fpr, fnr):
    scores = Scores(pos=[1, 2, 3], neg=[0, 0])
    roc_curve = roc_with_ci(scores, fpr=fpr, fnr=fnr, alpha=0.1)

    assert roc_curve.fnr.shape == roc_curve.fpr.shape
    assert roc_curve.fnr_ci.shape == roc_curve.fnr.shape + (2,)
    assert roc_curve.fpr_ci.shape == roc_curve.fpr.shape + (2,)


def test_roc_with_ci_2():
    """Test with more data"""
    rng = np.random.default_rng(seed=42)
    scores = Scores(
        pos=rng.normal(loc=0.1, size=100), neg=rng.normal(loc=-0.1, size=100)
    )
    roc_with_ci(scores, fpr=np.logspace(-2, -1))


def test_roc_with_ci_extra_points():
    scores = Scores(
        pos=np.linspace(0.0, 1.0, 100),
        neg=np.linspace(-0.2, 0.8, 100),
    )
    roc_curve = roc_with_ci(scores, fnr=[0.3, 0.4, 0.5])
    assert len(roc_curve.fnr) == 3 + ROC_CI_EXTRA_POINTS
