import numpy as np
import pandas as pd
import pytest

from score_analysis.cm import ConfusionMatrix


def test_from_predictions():
    labels = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    predictions = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    cm = ConfusionMatrix(labels=labels, predictions=predictions)

    assert isinstance(cm.classes, np.ndarray)
    np.testing.assert_equal(cm.classes, [0, 1, 2])
    assert isinstance(cm.matrix, np.ndarray)
    assert cm.matrix.dtype == int
    np.testing.assert_equal(cm.matrix, [[3, 0, 0], [0, 1, 2], [2, 1, 3]])


def test_from_predictions_classes():
    cm = ConfusionMatrix(labels=[0], predictions=[0], classes=[1, 0])

    np.testing.assert_equal(cm.classes, [1, 0])
    np.testing.assert_equal(cm.matrix, [[0, 0], [0, 1]])


def test_from_predictions_weights():
    cm = ConfusionMatrix(labels=[0, 1], predictions=[0, 2], weights=[0.5, 0.2])

    np.testing.assert_equal(cm.classes, [0, 1, 2])
    np.testing.assert_equal(cm.matrix, [[0.5, 0, 0], [0, 0, 0.2], [0, 0, 0]])
    assert cm.matrix.dtype == float


def test_from_nested_dicts():
    cm = ConfusionMatrix(
        matrix={0: {0: 3, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, 2: {0: 2, 1: 1, 2: 3}}
    )
    np.testing.assert_equal(cm.classes, [0, 1, 2])
    np.testing.assert_equal(cm.matrix, [[3, 0, 0], [0, 1, 2], [2, 1, 3]])


def test_from_nested_dicts_reorder():
    cm = ConfusionMatrix(
        matrix={0: {0: 3, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, 2: {0: 2, 1: 1, 2: 3}},
        classes=[2, 1, 0],
    )
    np.testing.assert_equal(cm.classes, [2, 1, 0])
    np.testing.assert_equal(cm.matrix, [[3, 1, 2], [2, 1, 0], [0, 0, 3]])


def test_from_dataframe():
    data = [[3, 0, 0], [0, 1, 2], [2, 1, 3]]
    cm = ConfusionMatrix(
        matrix=pd.DataFrame(data=data, index=[0, 1, 2], columns=[0, 1, 2])
    )
    np.testing.assert_equal(cm.classes, [0, 1, 2])
    np.testing.assert_equal(cm.matrix, data)


def test_from_dataframe_reorder():
    data = [[0, 2, 1], [3, 0, 0], [2, 3, 1]]
    cm = ConfusionMatrix(
        matrix=pd.DataFrame(data=data, index=[1, 0, 2], columns=[0, 2, 1]),
        classes=[2, 1, 0],
    )

    np.testing.assert_equal(cm.classes, [2, 1, 0])
    np.testing.assert_equal(cm.matrix, [[3, 1, 2], [2, 1, 0], [0, 0, 3]])


def test_from_nested_lists():
    data = [[3, 0, 0], [0, 1, 2], [2, 1, 3]]
    cm = ConfusionMatrix(matrix=data)
    np.testing.assert_equal(cm.classes, [0, 1, 2])
    np.testing.assert_equal(cm.matrix, data)

    cm = ConfusionMatrix(matrix=np.array(data), classes=["a", "b", "c"])
    np.testing.assert_equal(cm.classes, ["a", "b", "c"])
    np.testing.assert_equal(cm.matrix, data)


def test_binary():
    data = [[3, 2], [1, 4]]
    cm = ConfusionMatrix(matrix=data, binary=True)
    np.testing.assert_equal(cm.classes, [1, 0])

    cm = ConfusionMatrix(labels=[1, 0], predictions=[1, 0], binary=True)
    np.testing.assert_equal(cm.classes, [1, 0])


def test_invalid_init():
    # Initialize with matrix and additional arguments
    with pytest.raises(ValueError):
        ConfusionMatrix(labels=[], matrix=[[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        ConfusionMatrix(predictions=[], matrix=[[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        ConfusionMatrix(weights=[], matrix=[[1, 1], [1, 1]])

    # Initialize with missing arguments
    with pytest.raises(ValueError):
        ConfusionMatrix(labels=[1, 2], predictions=None)
    with pytest.raises(ValueError):
        ConfusionMatrix(labels=None, predictions=[1, 2])

    # Test invalid matrices and classes
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix=[2, 3])
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix=[[1, 2]])
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix=[[1]])
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix=[[1, 1], [1, 1]], classes=[1, 2, 3])
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix=[[1, 1], [1, 1]], classes=[1, 1])

    # Invalid parameters from dict
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix={0: {0: 2, 1: 2}, 1: {0: 2, 1: 2}}, classes=[1, 2])
    with pytest.raises(ValueError):
        ConfusionMatrix(matrix={0: {0: 2, 1: 2}, 1: {1: 2, 2: 2}}, classes=[0, 1])

    # Invalid parameters from dataframe
    with pytest.raises(ValueError):
        df = pd.DataFrame([[1, 1], [1, 1]], index=[0, 1], columns=[2, 3])
        ConfusionMatrix(matrix=df)
    with pytest.raises(ValueError):
        df = pd.DataFrame([[1, 1], [1, 1], [1, 1]], index=[0, 0, 1], columns=[0, 1])
        ConfusionMatrix(matrix=df)
    with pytest.raises(ValueError):
        df = pd.DataFrame([[1, 1, 1], [1, 1, 1]], index=[0, 1], columns=[0, 0, 1])
        ConfusionMatrix(matrix=df)
    with pytest.raises(ValueError):
        df = pd.DataFrame([[1, 1], [1, 1]], index=[0, 1], columns=[0, 1])
        ConfusionMatrix(matrix=df, classes=[1, 2])

    # Invalid weights vector
    with pytest.raises(ValueError):
        ConfusionMatrix(labels=[1, 2], predictions=[1, 2], weights=[1.0])

    # Invalid binary matrix
    with pytest.raises(ValueError):
        ConfusionMatrix(labels=[1], predictions=[2], classes=[0, 1, 2], binary=True)


def test_vectorized():
    data = np.array([[1, 0], [0, 1]])
    data = data[np.newaxis, np.newaxis, ...]
    assert data.shape == (1, 1, 2, 2)
    cm = ConfusionMatrix(matrix=data)

    sub_cm = cm[0, 0]
    assert isinstance(cm, ConfusionMatrix)
    assert sub_cm.matrix.shape == (2, 2)

    sub_cm = cm[0]
    assert isinstance(cm, ConfusionMatrix)
    assert sub_cm.matrix.shape == (1, 2, 2)


def test_basic_metrics():
    # fmt: off
    matrix = [
        [1, 2],
        [3, 4]
    ]
    # fmt: on

    cm = ConfusionMatrix(matrix=matrix)
    np.testing.assert_equal(cm.tp(), [1, 4])
    np.testing.assert_equal(cm.tn(), [4, 1])
    np.testing.assert_equal(cm.fp(), [3, 2])
    np.testing.assert_equal(cm.fn(), [2, 3])
    np.testing.assert_equal(cm.p(), [3, 7])
    np.testing.assert_equal(cm.n(), [7, 3])
    np.testing.assert_equal(cm.top(), [4, 6])
    np.testing.assert_equal(cm.ton(), [6, 4])
    np.testing.assert_equal(cm.pop(), 10)

    cm = ConfusionMatrix(matrix=matrix, binary=True)
    np.testing.assert_equal(cm.tp(), 1)
    np.testing.assert_equal(cm.tn(), 4)
    np.testing.assert_equal(cm.fp(), 3)
    np.testing.assert_equal(cm.fn(), 2)
    np.testing.assert_equal(cm.p(), 3)
    np.testing.assert_equal(cm.n(), 7)
    np.testing.assert_equal(cm.top(), 4)
    np.testing.assert_equal(cm.ton(), 6)


def test_accuracy():
    cm = ConfusionMatrix(matrix=[[1, 0], [1, 2]])
    assert cm.accuracy() == 0.75
    assert cm.error_rate() == 0.25


def test_one_vs_all_1():
    matrix = [[3, 0, 0], [0, 1, 2], [2, 1, 3]]
    cm = ConfusionMatrix(matrix=matrix)
    # fmt: off
    expected = [
        [[3, 0], [2, 7]],
        [[1, 2], [1, 8]],
        [[3, 3], [2, 4]],
    ]
    # fmt: on
    res = cm.one_vs_all()
    assert type(res) is ConfusionMatrix
    np.testing.assert_equal(res.matrix, expected)


def test_one_vs_all_2():
    cm = ConfusionMatrix(matrix=[[1, 2], [3, 4]])
    expected = [[[1, 2], [3, 4]], [[4, 3], [2, 1]]]
    np.testing.assert_equal(cm.one_vs_all().matrix, expected)


@pytest.mark.parametrize(
    "metric, expected",
    [
        ["tpr", [1.0 / 3.0, 0.8]],
        ["tnr", [0.8, 1.0 / 3.0]],
        ["fpr", [0.2, 2.0 / 3.0]],
        ["fnr", [2.0 / 3.0, 0.2]],
        ["ppv", [0.5, 2.0 / 3.0]],
        ["npv", [2.0 / 3.0, 0.5]],
        ["fdr", [0.5, 1.0 / 3.0]],
        ["for_", [1.0 / 3.0, 0.5]],
        ["class_accuracy", [0.625, 0.625]],
        ["class_error_rate", [0.375, 0.375]],
    ],
)
def test_per_class_metrics(metric, expected):
    # fmt: off
    matrix = [
        [1, 2],
        [1, 4],
    ]
    # fmt: on
    cm = ConfusionMatrix(matrix=matrix, classes=["a", "b"])

    # As array
    result = getattr(cm, metric)(as_dict=False)
    np.testing.assert_allclose(result, expected)

    # As dict
    result = getattr(cm, metric)(as_dict=True)
    np.testing.assert_allclose(result["a"], expected[0])
    np.testing.assert_allclose(result["b"], expected[1])


@pytest.mark.parametrize(
    "metric",
    [
        ConfusionMatrix.tpr_ci,
        ConfusionMatrix.tnr_ci,
        ConfusionMatrix.fpr_ci,
        ConfusionMatrix.fnr_ci,
    ],
)
def test_per_class_tpr_ci_etc(metric):
    alpha = 0.1
    m1 = [[3, 1], [5, 4]]
    m2 = [[4, 5], [1, 3]]

    cm = ConfusionMatrix(matrix=m1, classes=["a", "b"])
    bcm1 = ConfusionMatrix(matrix=m1, binary=True)
    bcm2 = ConfusionMatrix(matrix=m2, binary=True)
    res = metric(cm, alpha, as_dict=False)
    binary_res = np.stack([metric(bcm1, alpha), metric(bcm2, alpha)], axis=-2)
    np.testing.assert_allclose(res, binary_res)

    res = metric(cm, alpha, as_dict=True)
    np.testing.assert_allclose(res["a"], metric(bcm1, alpha))
    np.testing.assert_allclose(res["b"], metric(bcm2, alpha))


def test_binary_tpr_etc():
    # fmt: off
    matrix = [
        [1, 3],
        [2, 3]
    ]
    # fmt: on
    cm = ConfusionMatrix(matrix=matrix, binary=True)
    np.testing.assert_allclose(cm.tpr(), 0.25)
    np.testing.assert_allclose(cm.fnr(), 0.75)
    np.testing.assert_allclose(cm.tnr(), 0.6)
    np.testing.assert_allclose(cm.fpr(), 0.4)
    np.testing.assert_allclose(cm.ppv(), 1.0 / 3.0)
    np.testing.assert_allclose(cm.npv(), 0.5)
    np.testing.assert_allclose(cm.fdr(), 2.0 / 3.0)
    np.testing.assert_allclose(cm.for_(), 0.5)

    # Binary CMs cannot return per-class metrics as dicts.
    with pytest.raises(ValueError):
        cm.tpr(as_dict=True)


@pytest.mark.parametrize(
    "matrix, expected, metric",
    [
        [[[3, 1], [0, 0]], [0.32565535, 1.17434465], ConfusionMatrix.tpr_ci],
        [[[0, 0], [1, 3]], [0.32565535, 1.17434465], ConfusionMatrix.tnr_ci],
        [[[0, 0], [3, 1]], [0.32565535, 1.17434465], ConfusionMatrix.fpr_ci],
        [[[1, 3], [0, 0]], [0.32565535, 1.17434465], ConfusionMatrix.fnr_ci],
    ],
)
def test_binary_tpr_ci_etc(matrix, expected, metric):
    cm = ConfusionMatrix(matrix=matrix, binary=True)
    np.testing.assert_allclose(metric(cm, alpha=0.05), expected)


@pytest.mark.parametrize(
    "original, alias, kwargs",
    [
        [ConfusionMatrix.tpr, ConfusionMatrix.tar, {}],
        [ConfusionMatrix.fnr, ConfusionMatrix.frr, {}],
        [ConfusionMatrix.tnr, ConfusionMatrix.trr, {}],
        [ConfusionMatrix.fpr, ConfusionMatrix.far, {}],
        [ConfusionMatrix.tpr_ci, ConfusionMatrix.tar_ci, {"alpha": 0.1}],
        [ConfusionMatrix.fnr_ci, ConfusionMatrix.frr_ci, {"alpha": 0.1}],
        [ConfusionMatrix.tnr_ci, ConfusionMatrix.trr_ci, {"alpha": 0.1}],
        [ConfusionMatrix.fpr_ci, ConfusionMatrix.far_ci, {"alpha": 0.1}],
    ],
)
def test_aliases(original, alias, kwargs):
    cm = ConfusionMatrix(matrix=[[1, 3], [1, 10]], binary=True)
    np.testing.assert_equal(original(cm, **kwargs), alias(cm, **kwargs))

    cm = ConfusionMatrix(matrix=[[1, 3], [1, 10]], binary=False)
    np.testing.assert_equal(original(cm, **kwargs), alias(cm, **kwargs))
