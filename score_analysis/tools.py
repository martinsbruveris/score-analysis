"""
This module contains tools that can be useful in various applications but are somewhat
less fundemental than, e.g., the ConfusionMatrix class.

We make fewer promises about the backwards compatibility of the functions in this
module.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .scores import Scores


def roc_with_ci(
    scores: Scores,
    *,
    fpr: Optional[np.ndarray] = None,
    fnr: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    nb_samples: int = 1000,
    bootstrap_method: str = "bca",
    sampling_method: Union[str, Callable] = "replacement",
):
    """
    Convencience function to compute the ROC curve at given FPR or FNR values with a
    confidence interval.

    There are two ways to plot an ROC curve, depending on which metric (FPR or FNR) ir
    plotted on the x-axis. We expect the user to provide either FPR or FNR values (but
    not both). The provided values are plotted on the x-axis and used to set a threshold
    and then compute the other metric at those thresholds. The confidence band is
    computed using the union of confidence rectangles for both FNR and FPR.

    Args:
        scores: The Scores object whose ROC curve we compute.
        fpr: FPR values. If provided, will be set on x-axis
        fnr: FNR values. If provided, will be set on y-axis
        alpha: Significance level. In range (0, 1).
        nb_samples: Number of samples to bootstrap
        bootstrap_method: Method to compute the CI from the bootstrap samples. See the
            documentation of Scores.bootstrap_ci for details.
        sampling_method: Sampling method to create bootstrap sample. See the
            documentation of Scores.bootstrap_ci for details.

    Returns:
        (fpr, fnr, lower, upper) consisting of FPR and FNR values and the lower and
        upper bounds of the confidence interval.
    """
    if fpr is not None and fnr is not None:
        raise ValueError("Cannot provide both FPR and FNR.")
    if fpr is None and fnr is None:
        raise ValueError("Must provide at least one of FPR and FNR.")

    swap_axes = fpr is not None  # If True, we plot fpr on x-axis
    if swap_axes:
        scores = scores.swap()
        fnr = fpr
    # From here on we assume that we plot fnr on x-axis.

    # Use the given points to calculate initial thresholds
    fnr = np.asarray(fnr)
    threshold = scores.threshold_at_fnr(fnr)

    # We need to augment threshold to cover the full fnr and fpr ranges
    fpr_max = scores.fpr(threshold[0])
    fpr_high = np.linspace(1.0, fpr_max, num=20, endpoint=False)
    threshold_before = scores.threshold_at_fpr(fpr_high)

    fnr_max = scores.fnr(threshold[-1])
    fnr_high = np.linspace(1.0, fnr_max, num=20, endpoint=False)[::-1]
    threshold_after = scores.threshold_at_fnr(fnr_high)

    # Final set of thresholds
    threshold_all = np.concatenate([threshold_before, threshold, threshold_after])
    fpr_all = scores.fpr(threshold_all)
    fnr_all = scores.fnr(threshold_all)

    # Calculate bootstrap CI
    def _fpr_at_fnr(_scores: Scores, _fnr: np.ndarray = fnr_all) -> np.ndarray:
        _threshold = _scores.threshold_at_fnr(_fnr)
        _fpr = _scores.fpr(_threshold)
        return _fpr

    def _fnr_at_fpr(_scores: Scores, _fpr: np.ndarray = fpr_all) -> np.ndarray:
        _threshold = _scores.threshold_at_fpr(_fpr)
        _fnr = _scores.fnr(_threshold)
        return _fnr

    # Compute CIs in each direction
    fnr_ci = scores.bootstrap_ci(
        metric=_fnr_at_fpr,
        alpha=alpha,
        nb_samples=nb_samples,
        bootstrap_method=bootstrap_method,
        sampling_method=sampling_method,
    )
    fpr_ci = scores.bootstrap_ci(
        metric=_fpr_at_fnr,
        alpha=alpha,
        nb_samples=nb_samples,
        bootstrap_method=bootstrap_method,
        sampling_method=sampling_method,
    )
    lower, upper = _aggregate_rectangles(fnr, fnr_ci, fpr_ci)

    fpr = scores.fpr(threshold)
    if swap_axes:
        fnr, fpr = fpr, fnr

    return fpr, fnr, lower, upper


def _aggregate_rectangles(
    x: np.ndarray, dxp: np.ndarray, dyp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function aggregates rectangles into one continous band and returns
    the lower and upper limits of the band at the points given by x.

    Args:
        x: Array of shape (N,) of points where to evaluate the band
        dxp: Array of shape (M, 2) with x-limits of rectangles
        dyp: Array of shape (M, 2) with y-limits of rectangles

    Returns:
        lower, upper: Arrays of shape (N,) where
            * lower[i] is the minimum of dyp[j, 0] for which x[i] lies in the interval
              defined by dxp[j]
            * upper[i] is the maximum of dyp[j, 1] with the same condition

            The values lower[i] and upper[i] are undefined if no rectangle covers x[i].
    """
    x = np.asarray(x)
    dxp = np.asarray(dxp)
    dyp = np.asarray(dyp)

    lower = np.empty_like(x)
    upper = np.empty_like(x)
    for j in range(len(x)):
        inside = (dxp[:, 0] <= x[j]) & (x[j] <= dxp[:, 1])
        if np.sum(inside) > 0:
            lower[j] = np.min(dyp[inside, 0])
            upper[j] = np.max(dyp[inside, 1])
    return lower, upper
