"""
This module contains internal utility functions.
"""
import numpy as np
from scipy import stats


def binomial_ci(count: np.ndarray, nobs: np.ndarray, alpha: float = 0.05):
    """
    Confidence interval for binomial proportions.

    Args:
        count: Number of successes of shape (...)
        nobs: Number of trials of shape (...)
        alpha: Significance level. In range (0, 1)

    Returns:
        np.ndarray of shape (..., 2). Lower and upper limits of confidence interval
        with coverage 1-alpha.
    """
    nans = np.full_like(count, np.nan, dtype=float)
    p = np.divide(count, nobs, out=nans, where=nobs != 0)

    nans = np.full_like(count, np.nan, dtype=float)
    std = np.divide(p * (1 - p), nobs, out=nans, where=nobs != 0)
    std = np.sqrt(std)

    dist = stats.norm.isf(alpha / 2.0) * std
    ci = np.stack([p - dist, p + dist], axis=-1)
    return ci
