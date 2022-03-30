from typing import Union

import numpy as np


def accuracy(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Accuracy for confusion matrices.

    Args:
        matrix: Array of shape (..., N, N).

    Returns:
        Array of shape (...). Returns NaNs in case of zero matrix.
    """
    # Correctly classified elements are along the diagonal
    correct = np.diagonal(matrix, axis1=-1, axis2=-2)
    correct = np.sum(correct, axis=-1)
    total = np.sum(matrix, axis=(-1, -2))
    res = np.divide(
        correct, total, out=np.full_like(correct, np.nan, dtype=float), where=total != 0
    )
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def tpr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Positive Rate for binary confusion matrices.

    Formula:
        TPR = TP / (TP + FN) = TP / P

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with positive
        condition.
    """

    tp = matrix[..., 0, 0]
    p = matrix[..., 0, 0] + matrix[..., 0, 1]
    res = np.divide(tp, p, out=np.full_like(tp, np.nan, dtype=float), where=p != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def fnr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Negative Rate for binary confusion matrices.

    Formula:
        FNR = FN / (TP + FN) = FN / P

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with positive
        condition.
    """

    fn = matrix[..., 0, 1]
    p = matrix[..., 0, 0] + matrix[..., 0, 1]
    res = np.divide(fn, p, out=np.full_like(fn, np.nan, dtype=float), where=p != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def tnr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Negative Rate for binary confusion matrices.

    Formula:
        TNR = TN / (FP + TN) = TN / N

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with positive
        condition.
    """

    tn = matrix[..., 1, 1]
    n = matrix[..., 1, 0] + matrix[..., 1, 1]
    res = np.divide(tn, n, out=np.full_like(tn, np.nan, dtype=float), where=n != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def fpr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Positive Rate for binary confusion matrices.

    Formula:
        FPR = FP / (FP + TN) = FP / N

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with positive
        condition.
    """

    fp = matrix[..., 1, 0]
    n = matrix[..., 1, 0] + matrix[..., 1, 1]
    res = np.divide(fp, n, out=np.full_like(fp, np.nan, dtype=float), where=n != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res
