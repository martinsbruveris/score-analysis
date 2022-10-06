"""
This module contains the fundamental metrics computations. The ConfusionMatrix class
relies on the implementations in this module.
"""

from typing import Union

import numpy as np

from .utils import binomial_ci


def tp(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Positives for binary confusion matrices. Test detects the condition while the
    condition is present.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 0, 0]


def tn(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Negatives for binary confusion matrices. Test does not detect the condition and
    the condition is absent.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 1, 1]


def fp(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Positives for binary confusion matrices. Test detects the condition while the
    condition is absent.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 1, 0]


def fn(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Negatives for binary confusion matrices. Test does not detect the condition
    while the condition is present.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 0, 1]


def p(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Condition Positive for binary confusion matrices. Number of samples with condition
    positive.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 0, 0] + matrix[..., 0, 1]


def n(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Condition Negative for binary confusion matrices. Number of samples with condition
    negative.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 1, 0] + matrix[..., 1, 1]


def top(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Test Outcome Positive. Number of samples where test detects condition.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 0, 0] + matrix[..., 1, 0]


def ton(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Test Outcome Negative. Number of samples where test does not detect condition.

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    return matrix[..., 0, 1] + matrix[..., 1, 1]


def pop(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Total population for confusion matrices.

    Args:
        matrix: Array of shape (..., N, N).

    Returns:
        Array of shape (...).
    """
    return np.sum(matrix, axis=(-1, -2))


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


def error_rate(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Error rate for confusion matrices.

    Formula::

        Error Rate = 1 - Accuracy

    Args:
        matrix: Array of shape (..., N, N).

    Returns:
        Array of shape (...). Returns NaNs in case of zero matrix.
    """
    return 1 - accuracy(matrix)


def tpr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Positive Rate for binary confusion matrices.

    Formula::

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


def tnr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Negative Rate for binary confusion matrices.

    Formula::

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

    Formula::

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


def fnr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Negative Rate for binary confusion matrices.

    Formula::

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


# Aliases for within Onfido use.
def tar(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Acceptance Rate. Alias for :func:`tpr`.
    """
    return tpr(matrix)


def frr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Rejection Rate. Alias for :func:`fnr`.
    """
    return fnr(matrix)


def trr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    True Rejection Rate. Alias for :func:`tnr`.
    """
    return tnr(matrix)


def far(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Acceptance Rate. Alias for :func:`fpr`.
    """
    return fpr(matrix)


def tpr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence inferval for the True Positive Rate for binary confusion matrices.

    Args:
        matrix: Array of shape (..., 2, 2)
        alpha: Significance level. In range (0, 1).

    Returns:
        Array of shape (..., 2). Lower and upper limits of CI with coverage 1-alpha.
    """
    return binomial_ci(count=tp(matrix), nobs=p(matrix), alpha=alpha)


def tnr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence inferval for the True Negative Rate for binary confusion matrices.

    Args:
        matrix: Array of shape (..., 2, 2)
        alpha: Significance level. In range (0, 1).

    Returns:
        Array of shape (..., 2). Lower and upper limits of CI with coverage 1-alpha.
    """
    return binomial_ci(count=tn(matrix), nobs=n(matrix), alpha=alpha)


def fpr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence inferval for the False Positive Rate for binary confusion matrices.

    Args:
        matrix: Array of shape (..., 2, 2)
        alpha: Significance level. In range (0, 1).

    Returns:
        Array of shape (..., 2). Lower and upper limits of CI with coverage 1-alpha.
    """
    return binomial_ci(count=fp(matrix), nobs=n(matrix), alpha=alpha)


def fnr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence inferval for the False Negative Rate for binary confusion matrices.

    Args:
        matrix: Array of shape (..., 2, 2)
        alpha: Significance level. In range (0, 1).

    Returns:
        Array of shape (..., 2). Lower and upper limits of CI with coverage 1-alpha.
    """
    return binomial_ci(count=fn(matrix), nobs=p(matrix), alpha=alpha)


# Aliases for within Onfido use.
def tar_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence interval for the True Acceptance Rate. Alias for :func:`tpr_ci`.
    """
    return tpr_ci(matrix, alpha)


def frr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence interval for the False Rejection Rate. Alias for :func:`fnr_ci`.
    """
    return fnr_ci(matrix, alpha)


def trr_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence interval for the True Rejection Rate. Alias for :func:`tnr_ci`.
    """
    return tnr_ci(matrix, alpha)


def far_ci(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Confidence interval for the False Acceptance Rate. Alias for :func:`fpr_ci`.
    """
    return fpr_ci(matrix, alpha)


def topr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Test Outcome Positive Rate. Proportion of samples where test detects condition.

    Formula::

        TOPR = TOP / N = (TPR + FPR) / (TPR + FPR + TNR + FNR)

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    tops = top(matrix)
    pops = pop(matrix)
    res = np.divide(
        tops, pops, out=np.full_like(tops, np.nan, dtype=float), where=n != 0
    )
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def tonr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Test Outcome Negative Rate.
    Proportion of samples where test does not detect condition.

    Formula::

        TONR = TON / N = (TNR + FNR) / (TPR + FPR + TNR + FNR)

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...).
    """
    tons = ton(matrix)
    pops = pop(matrix)
    res = np.divide(
        tons, pops, out=np.full_like(tons, np.nan, dtype=float), where=n != 0
    )
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


# Aliases for within Onfido use.
def acceptance_rate(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Acceptance Rate. Alias for :func:`topr`.
    """
    return topr(matrix)


def rejection_rate(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Rejection Rate. Alias for :func:`tonr`.
    """
    return tonr(matrix)


def ppv(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Positive Predictive Value for binary confusion matrices.

    Formula::

        PPV = TP / (TP + FP) = TP / TOP

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with prediction
        positive.
    """
    tp = matrix[..., 0, 0]
    top = matrix[..., 0, 0] + matrix[..., 1, 0]
    res = np.divide(tp, top, out=np.full_like(tp, np.nan, dtype=float), where=top != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def npv(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    Negative Predictive Value for binary confusion matrices.

    Formula::

        NPV = TN / (TN + FN) = TN / TON

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with prediction
        positive.
    """
    tn = matrix[..., 1, 1]
    ton = matrix[..., 1, 1] + matrix[..., 0, 1]
    res = np.divide(tn, ton, out=np.full_like(tn, np.nan, dtype=float), where=ton != 0)
    res = res.item() if res.ndim == 0 else res  # Reduce to scalar
    return res


def fdr(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Discovery Rate for binary confusion matrices.

    Formula::

        FDR = FP / (TP + FP) = FP / TOP = 1 - PPV

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with prediction
        positive.
    """
    return 1 - ppv(matrix)


def for_(matrix: np.ndarray) -> Union[np.ndarray, float]:
    """
    False Omission Rate for binary confusion matrices.

    Formula::

        FOR = FN / (TN + FN) = FN / TON = 1 - NPV

    Args:
        matrix: Array of shape (..., 2, 2).

    Returns:
        Array of shape (...). Returns NaNs in case of no elements with prediction
        negative.
    """
    return 1 - npv(matrix)
