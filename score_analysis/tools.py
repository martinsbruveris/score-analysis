"""
This module contains tools that can be useful in various applications but are somewhat
less fundemental than, e.g., the ConfusionMatrix class.
"""
from functools import partial
from typing import Optional

import numpy as np

from .scores import Scores


def roc_with_ci(
    scores: Scores,
    *,
    fpr: Optional[np.ndarray] = None,
    fnr: Optional[np.ndarray] = None,
    method: str = "pessimist",
    alpha: float = 0.05,
):
    """
    Convencience function to compute the ROC curve at given FPR or FNR values with a
    confidence interval.

    There are two ways to plot an ROC curve, depending on which metric (FPR or FNR) ir
    plotted on the x-axis. We expect the user to provide either FPR or FNR values (but
    not both). The provided values are plotted on the x-axis and used to set a threshold
    and then compute the other metric at those thresholds. The confidence interval is
    computed for the y-axis metric.

    We provide three methods for calculating confidence intervals.

    * "bootstrap" will use the Scores.bootstrap_ci method to calculate the uncertainty
      of the FPR -> threshold -> FNR (or other way round) mapping.
    * "binomial" will use the ConfusionMatrix.fnr_ci method to calculate the uncertainty
      of FNR at a fixed threshold assuming a binomial distribution of false negatives
      among all positives.
    * "pessimist" will take the union of the above two methods, since "binomial" leads
      to too small intervals in the high FNR range, while "bootstrap" is too confident
      in the low FNR range. This is the recommended method.

    Args:
        scores: The Scores object whose ROC curve we compute.
        fpr: FPR values. If provided, will be set on x-axis
        fnr: FNR values. If provided, will be set on y-axis
        method: Method for calculating confidence intervals. One of "pessimist",
            "bootstrap" or "binomial".
        alpha: Significance level. In range (0, 1).

    Returns:
        (fpr, fnr, lower, upper) consisting of FPR and FNR values and the lower and
        upper bounds of the confidence interval.
    """
    if fpr is not None and fnr is not None:
        raise ValueError("Cannot provide both FPR and FNR.")
    if fpr is None and fnr is None:
        raise ValueError("Must provide at least one of FPR and FNR.")
    if method not in {"bootstrap", "binomial", "pessimist"}:
        raise ValueError(f"Unknown method {method} for computing CI.")

    # Calculate bootstrap CI
    def _fpr_at_fnr(_scores: Scores, _fnr: np.ndarray) -> np.ndarray:
        _threshold = _scores.threshold_at_fnr(_fnr)
        _fpr = _scores.fpr(_threshold)
        return _fpr

    def _fnr_at_fpr(_scores: Scores, _fpr: np.ndarray) -> np.ndarray:
        _threshold = _scores.threshold_at_fpr(_fpr)
        _fnr = _scores.fnr(_threshold)
        return _fnr

    if fnr is not None:
        fnr = np.asarray(fnr)
        metric = partial(_fpr_at_fnr, _fnr=fnr)
    else:
        fpr = np.asarray(fpr)
        metric = partial(_fnr_at_fpr, _fpr=fpr)

    if method in {"bootstrap", "pessimist"}:
        bootstrap_ci = scores.bootstrap_ci(
            metric=metric,
            alpha=alpha,
            nb_samples=1000,
            method="replacement",
        )
    else:
        bootstrap_ci = None

    # Calculate binomial CI
    if method in {"binomial", "pessimist"}:
        if fnr is not None:
            threshold = scores.threshold_at_fnr(fnr)
            cm = scores.cm(threshold)
            binomial_ci = cm.fpr_ci(alpha=alpha)
        else:
            threshold = scores.threshold_at_fpr(fpr)
            cm = scores.cm(threshold)
            binomial_ci = cm.fnr_ci(alpha=alpha)
    else:
        binomial_ci = None

    # Based on method select right one
    if method == "bootstrap":
        lower = bootstrap_ci[:, 0]
        upper = bootstrap_ci[:, 1]
    elif method == "binomial":
        lower = binomial_ci[:, 0]
        upper = binomial_ci[:, 1]
    else:
        # Take the worst ones
        lower = np.minimum(bootstrap_ci[:, 0], binomial_ci[:, 0])
        upper = np.maximum(bootstrap_ci[:, 1], binomial_ci[:, 1])

    # Calculate the y-axis values
    if fnr is not None:
        fpr = _fpr_at_fnr(scores, fnr)
    else:
        fnr = _fnr_at_fpr(scores, fpr)

    return fpr, fnr, lower, upper
