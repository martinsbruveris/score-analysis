import numpy as np
import pytest

from score_analysis import metrics


@pytest.mark.parametrize(
    "metric, expected",
    [
        [metrics.tp, 1],
        [metrics.tn, 4],
        [metrics.fp, 3],
        [metrics.fn, 2],
        [metrics.p, 3],
        [metrics.n, 7],
        [metrics.top, 4],
        [metrics.ton, 6],
        [metrics.pop, 10],
    ],
)
def test_basic_metrics(metric, expected):
    # fmt: off
    matrix = [
        [1, 2],
        [3, 4]
    ]
    # fmt: on
    np.testing.assert_equal(metric(np.asarray(matrix)), expected)
    # Test vectorized version
    np.testing.assert_equal(metric(np.asarray([matrix, matrix])), [expected, expected])


@pytest.mark.parametrize(
    "matrix, expected, isscalar, func",
    [
        # Accuracy
        [[[1, 4], [3, 2]], 0.3, True, metrics.accuracy],
        [[[1, 3, 0], [0, 2, 1], [1, 1, 1]], 0.4, True, metrics.accuracy],
        [[[0, 0], [0, 0]], np.nan, True, metrics.accuracy],
        [[[[1, 4], [3, 2]]], [0.3], False, metrics.accuracy],
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.accuracy],
        # Error rate
        [[[1, 4], [3, 2]], 0.7, True, metrics.error_rate],
        # TPR
        [[[1, 3], [3, 4]], 0.25, True, metrics.tpr],  # Regular computation
        [[[2, 3], [0, 0]], 0.4, True, metrics.tpr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.tpr],  # Nans
        [[[[1, 3], [0, 0]]], [0.25], False, metrics.tpr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.tpr],  # Vectorized nans
        # TNR
        [[[3, 4], [1, 3]], 0.75, True, metrics.tnr],  # Regular computation
        [[[0, 0], [2, 3]], 0.6, True, metrics.tnr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.tnr],  # Nans
        [[[[0, 0], [1, 3]]], [0.75], False, metrics.tnr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.tnr],  # Vectorized nans
        # FNR
        [[[1, 3], [3, 4]], 0.75, True, metrics.fnr],  # Regular computation
        [[[2, 3], [0, 0]], 0.6, True, metrics.fnr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.fnr],  # Nans
        [[[[1, 3], [0, 0]]], [0.75], False, metrics.fnr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.fnr],  # Vectorized nans
        # FPR
        [[[3, 4], [1, 3]], 0.25, True, metrics.fpr],  # Regular computation
        [[[0, 0], [2, 3]], 0.4, True, metrics.fpr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.fpr],  # Nans
        [[[[0, 0], [1, 3]]], [0.25], False, metrics.fpr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.fpr],  # Vectorized nans
        # TOPR
        [[[1, 3], [2, 6]], 0.25, True, metrics.topr],  # Regular computation
        [[[2, 0], [3, 0]], 1.0, True, metrics.topr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.topr],  # Nans
        [[[[0, 5], [0, 4]]], [0.0], False, metrics.topr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.topr],  # Vectorized nans
        # TONR
        [[[2, 4], [1, 5]], 0.75, True, metrics.tonr],  # Regular computation
        [[[0, 2], [0, 3]], 1.0, True, metrics.tonr],  # Zeros in other row
        [[[0, 0], [0, 0]], np.nan, True, metrics.tonr],  # Nans
        [[[[5, 0], [4, 0]]], [0.0], False, metrics.tonr],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.tonr],  # Vectorized nans
        # PPV
        [[[3, 4], [1, 3]], 0.75, True, metrics.ppv],  # Regular computation
        [[[3, 0], [1, 0]], 0.75, True, metrics.ppv],  # Zeros in other column
        [[[0, 0], [0, 0]], np.nan, True, metrics.ppv],  # Nans
        [[[[3, 0], [1, 0]]], [0.75], False, metrics.ppv],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.ppv],  # Vectorized nans
        # NPV
        [[[3, 4], [1, 1]], 0.2, True, metrics.npv],  # Regular computation
        [[[0, 4], [0, 1]], 0.2, True, metrics.npv],  # Zeros in other column
        [[[0, 0], [0, 0]], np.nan, True, metrics.npv],  # Nans
        [[[[0, 4], [0, 1]]], [0.2], False, metrics.npv],  # Vectorized
        [[[[[0, 0], [0, 0]]]], [[np.nan]], False, metrics.npv],  # Vectorized nans
        # FDR
        [[[3, 4], [1, 3]], 0.25, True, metrics.fdr],  # Regular computation
        # FOR
        [[[3, 4], [1, 1]], 0.8, True, metrics.for_],  # Regular computation
    ],
)
def test_metric(matrix, expected, isscalar, func):
    matrix = np.asarray(matrix)
    expected = np.asarray(expected)
    result = func(matrix)
    np.testing.assert_equal(result, expected)
    assert np.isscalar(result) == isscalar


@pytest.mark.parametrize(
    "matrix, expected, func, alpha",
    [
        # TPR CI
        # Regular computation
        [[[3, 1], [3, 4]], [0.32565535, 1.17434465], metrics.tpr_ci, 0.05],
        [[[3, 1], [3, 4]], [0.39387874, 1.10612126], metrics.tpr_ci, 0.1],
        [[[0, 0], [0, 0]], [np.nan, np.nan], metrics.tpr_ci, 0.05],  # Nans
        # Vectorized
        [[[[3, 1], [0, 0]]], [[0.32565535, 1.17434465]], metrics.tpr_ci, 0.05],
        [[[[[0, 0], [0, 0]]]], [[[np.nan, np.nan]]], metrics.tpr_ci, 0.05],
        # TNR CI
        # Regular computation
        [[[3, 4], [1, 3]], [0.32565535, 1.17434465], metrics.tnr_ci, 0.05],
        [[[3, 4], [1, 3]], [0.39387874, 1.10612126], metrics.tnr_ci, 0.1],
        [[[0, 0], [0, 0]], [np.nan, np.nan], metrics.tnr_ci, 0.05],  # Nans
        # Vectorized
        [[[[0, 0], [1, 3]]], [[0.32565535, 1.17434465]], metrics.tnr_ci, 0.05],
        [[[[[0, 0], [0, 0]]]], [[[np.nan, np.nan]]], metrics.tnr_ci, 0.05],
        # FPR CI
        # Regular computation
        [[[3, 4], [3, 1]], [0.32565535, 1.17434465], metrics.fpr_ci, 0.05],
        [[[3, 4], [3, 1]], [0.39387874, 1.10612126], metrics.fpr_ci, 0.1],
        [[[0, 0], [0, 0]], [np.nan, np.nan], metrics.fpr_ci, 0.05],  # Nans
        # Vectorized
        [[[[0, 0], [3, 1]]], [[0.32565535, 1.17434465]], metrics.fpr_ci, 0.05],
        [[[[[0, 0], [0, 0]]]], [[[np.nan, np.nan]]], metrics.fpr_ci, 0.05],
        # FNR CI
        # Regular computation
        [[[1, 3], [3, 4]], [0.32565535, 1.17434465], metrics.fnr_ci, 0.05],
        [[[1, 3], [3, 4]], [0.39387874, 1.10612126], metrics.fnr_ci, 0.1],
        [[[0, 0], [0, 0]], [np.nan, np.nan], metrics.fnr_ci, 0.05],  # Nans
        # Vectorized
        [[[[1, 3], [0, 0]]], [[0.32565535, 1.17434465]], metrics.fnr_ci, 0.05],
        [[[[[0, 0], [0, 0]]]], [[[np.nan, np.nan]]], metrics.fnr_ci, 0.05],
    ],
)
def test_metric_ci(matrix, expected, func, alpha):
    matrix = np.asarray(matrix)
    expected = np.asarray(expected)
    result = func(matrix, alpha)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "original, alias, args",
    [
        [metrics.tpr, metrics.tar, [np.array([[1, 0], [3, 2]])]],
        [metrics.fnr, metrics.frr, [np.array([[1, 0], [3, 2]])]],
        [metrics.tnr, metrics.trr, [np.array([[1, 0], [3, 2]])]],
        [metrics.fpr, metrics.far, [np.array([[1, 0], [3, 2]])]],
        [metrics.topr, metrics.acceptance_rate, [np.array([[1, 0], [3, 2]])]],
        [metrics.tonr, metrics.rejection_rate, [np.array([[1, 0], [3, 2]])]],
        [metrics.tpr_ci, metrics.tar_ci, [np.array([[1, 0], [3, 2]]), 0.1]],
        [metrics.fnr_ci, metrics.frr_ci, [np.array([[1, 0], [3, 2]]), 0.1]],
        [metrics.tnr_ci, metrics.trr_ci, [np.array([[1, 0], [3, 2]]), 0.1]],
        [metrics.fpr_ci, metrics.far_ci, [np.array([[1, 0], [3, 2]]), 0.1]],
    ],
)
def test_aliases(original, alias, args):
    np.testing.assert_equal(original(*args), alias(*args))
