import numpy as np
import pandas as pd

from score_analysis.cm import ConfusionMatrix, BinaryConfusionMatrix


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
        classes=[2, 1, 0]
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
        classes=[2, 1, 0]
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


def test_accuracy():
    cm = ConfusionMatrix(matrix=[[1, 0], [1, 2]])
    assert cm.accuracy() == 0.75


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
    np.testing.assert_equal(cm.one_vs_all().matrix, expected)


def test_one_vs_all_2():
    cm = ConfusionMatrix(matrix=[[1, 2], [3, 4]])
    expected = [[[1, 2], [3, 4]], [[4, 3], [2, 1]]]
    np.testing.assert_equal(cm.one_vs_all().matrix, expected)


def test_binary_tpr_etc():
    cm = BinaryConfusionMatrix(matrix=[[1, 3], [2, 3]])
    assert cm.tpr() == 0.25
    assert cm.fnr() == 0.75
    assert cm.tnr() == 0.6
    assert cm.fpr() == 0.4
