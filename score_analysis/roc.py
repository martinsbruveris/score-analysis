"""
This module contains ROC curve calculations, including confidence bands for ROC curves.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .scores import DEFAULT_BOOTSTRAP_CONFIG, BootstrapConfig, Scores


@dataclass
class ROCCurve:
    fnr: np.ndarray
    fpr: np.ndarray
    fnr_ci: Optional[np.ndarray] = None
    fpr_ci: Optional[np.ndarray] = None

    @property
    def tpr(self):
        return 1.0 - self.fnr

    @property
    def tnr(self):
        return 1.0 - self.fpr

    @property
    def tpr_ci(self):
        return None if self.fnr_ci is None else np.copy(1.0 - self.fnr_ci[..., ::-1])

    @property
    def tnr_ci(self):
        return None if self.fpr_ci is None else np.copy(1.0 - self.fpr_ci[..., ::-1])


def roc(
    scores: Scores,
    *,
    fnr: Optional[np.ndarray] = None,
    fpr: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    nb_points: int = 100,
) -> ROCCurve:
    """Compute the ROC curve at the given FNR, FPR or threshold values."""
    if thresholds is None:
        thresholds = np.zeros(shape=(0,))
    if fnr is not None:
        fnr_thresholds = scores.threshold_at_fnr(fnr)
        thresholds = np.concatenate([thresholds, fnr_thresholds])
    if fpr is not None:
        fpr_thresholds = scores.threshold_at_fpr(fpr)
        thresholds = np.concatenate([thresholds, fpr_thresholds])
    if len(thresholds) == 0:
        nb_fnr_points = nb_points // 2
        nb_fpr_points = nb_points - nb_fnr_points
        default_fnr = np.linspace(0.0, 1.0, nb_fnr_points, endpoint=True)
        fnr_thresholds = scores.threshold_at_fnr(default_fnr)

        default_fpr = np.linspace(0.0, 1.0, nb_fpr_points, endpoint=True)
        fpr_thresholds = scores.threshold_at_fpr(default_fpr)
        thresholds = np.concatenate([fnr_thresholds, fpr_thresholds])

        thresholds = np.sort(thresholds)

    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

    return ROCCurve(fnr=fnr, fpr=fpr)


def roc_with_ci(
    scores: Scores,
    *,
    fnr: Optional[np.ndarray] = None,
    fpr: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    nb_points: Optional[int] = None,
    alpha: float = 0.05,
    config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
) -> ROCCurve:
    """
    Function to compute the confidence band around a ROC curve.

    There are two ways to plot an ROC curve, depending on which metric (FPR or FNR) ir
    plotted on the x-axis. We expect the user to provide either FPR or FNR values (but
    not both). The provided values are plotted on the x-axis and used to set a threshold
    and then compute the other metric at those thresholds. The confidence band is
    computed using the union of confidence rectangles for both FNR and FPR.

    Args:
        scores: The Scores object whose ROC curve we compute.
        fnr: Optional FNR support points on the ROC curve.
        fpr: Optional FPR support points on the ROC curve.
        thresholds: Optional threshold support points on the ROC curve.
        nb_points: If at least one of fnr, fpr and threshold is provided, this is the
            additional number of support points beyond the range of the specified
            points. If none of fnr, fpr or threshold are provided, we use this number
            of support points linearly spaced along the FNR and FPR axes.
        alpha: Significance level. In range (0, 1).
        config: Bootstrap config.

    Returns:
        ROCCurve object with point values for the ROC curve and the lower and upper
        bounds of the confidence band values for both metric values.
    """
    thresholds = _find_support_thresholds(
        scores=scores, fnr=fnr, fpr=fpr, thresholds=thresholds, nb_points=nb_points
    )
    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

    # Calculate bootstrap CI
    def _metric(_scores: Scores):
        _fnr = _scores.fnr(_scores.threshold_at_fpr(fpr))
        _fpr = _scores.fpr(_scores.threshold_at_fnr(fnr))
        return np.stack([_fnr, _fpr], axis=0)

    joint_ci = scores.bootstrap_ci(metric=_metric, alpha=alpha, config=config)
    fnr_ci = joint_ci[0]
    fpr_ci = joint_ci[1]

    # Rule-of-three correction for FNR and FPR being 0. or 1.
    fnr_ci = _apply_rule_of_three(p=fnr, ci=fnr_ci, alpha=alpha, n=len(scores.pos))
    fpr_ci = _apply_rule_of_three(p=fpr, ci=fpr_ci, alpha=alpha, n=len(scores.neg))

    # Here the magic happens, and we aggregate 1D CIs to a 2D confidence band.
    fpr_band = _aggregate_rectangles(fnr, fnr_ci, fpr_ci)
    fnr_band = _aggregate_rectangles(fpr, fpr_ci, fnr_ci)

    return ROCCurve(fnr=fnr, fpr=fpr, fnr_ci=fnr_band, fpr_ci=fpr_band)


def _find_support_thresholds(
    scores: Scores,
    fnr: Optional[np.ndarray],
    fpr: Optional[np.ndarray],
    thresholds: Optional[np.ndarray],
    nb_points: Optional[int],
) -> np.ndarray:
    if fnr is None and fpr is None and thresholds is None:
        default_nb_points = 100
    else:
        default_nb_points = 20
    nb_points = nb_points or default_nb_points

    # We subtract 4 to make space for thresholds beyond the pos and neg score limits;
    # see np.nextafter calls below. These are number of points in addition to those
    # provided by the user in fnr and fpr parameters.
    nb_fnr_points = (nb_points - 4) // 2
    nb_fpr_points = nb_points - 4 - nb_fnr_points

    fnr_init = _combine_support_points(fnr, nb_fnr_points)
    fnr_thresholds = np.concatenate(
        [
            scores.threshold_at_fnr(fnr_init),
            # Add thresholds that are smaller and larger than all scores
            [np.nextafter(scores.pos[0], -np.inf)],
            [np.nextafter(scores.pos[-1], np.inf)],
        ]
    )

    fpr_init = _combine_support_points(fpr, nb_fpr_points)
    fpr_thresholds = np.concatenate(
        [
            scores.threshold_at_fpr(fpr_init),
            # Add thresholds that are smaller and larger than all scores
            [np.nextafter(scores.neg[0], -np.inf)],
            [np.nextafter(scores.neg[-1], np.inf)],
        ]
    )

    if thresholds is None:
        thresholds = np.zeros(shape=(0,))
    thresholds = np.concatenate([fnr_thresholds, fpr_thresholds, thresholds])
    thresholds = np.sort(thresholds)
    if scores.score_class == "neg":
        thresholds = thresholds[::-1]  # We want FNR to be increasing

    return thresholds


def _combine_support_points(x: Optional[np.ndarray], nb_points: int) -> np.ndarray:
    if x is None:
        return np.linspace(0.0, 1.0, num=nb_points, endpoint=True)

    x = np.sort(x)
    nb_before = nb_points // 2
    nb_after = nb_points - nb_before
    if x[0] > 0.0:
        x_before = np.linspace(0.0, x[0], num=nb_before, endpoint=False)
        x = np.concatenate([x_before, x])
    if x[-1] < 1.0:
        x_after = np.linspace(1.0, x[-1], num=nb_after, endpoint=False)
        x = np.concatenate([x, x_after])
    return x


def _apply_rule_of_three(
    p: np.ndarray, ci: np.ndarray, alpha: float, n: int
) -> np.ndarray:
    """
    When FNR or FPR is 0. or 1., we cannot use bootstrapping to estimate uncertainty
    as we don't have any samples to measure probabilities smaller than 1/n. In these
    cases, we use the generalised rule-of-three approximation (when alpha=0.05, then
    the confidence interval is approx. (0, 3/n).

    See: https://en.wikipedia.org/wiki/Rule_of_three_(statistics)
    """
    lower_correction = np.array([[0.0, 1 - math.pow(alpha, 1 / n)]])
    upper_correction = np.array([[math.pow(alpha, 1 / n), 1.0]])

    ci = np.where(p[:, np.newaxis] < 1.0 / n, lower_correction, ci)
    ci = np.where(p[:, np.newaxis] > (n - 1) / n, upper_correction, ci)
    return ci


def _aggregate_rectangles(
    x: np.ndarray, dxp: np.ndarray, dyp: np.ndarray
) -> np.ndarray:
    """
    Function aggregates rectangles into one continuous band and returns
    the lower and upper limits of the band at the points given by x.

    Args:
        x: Array of shape (N, ) of points where to evaluate the band
        dxp: Array of shape (M, 2) with x-limits of rectangles
        dyp: Array of shape (M, 2) with y-limits of rectangles

    Returns:
        ci: Array of shape (N, 2) where
            * ci[:, 0] is the minimum of dyp[j, 0] for which x[i] lies in the interval
              defined by dxp[j]
            * ci[:, 1] is the maximum of dyp[j, 1] with the same condition

            The value of ci is undefined if there is no rectangle covering a point.
    """
    x = np.asarray(x)
    dxp = np.asarray(dxp)
    dyp = np.asarray(dyp)

    # The variation at x is at least as large as given by the vertical extent of the
    # rectangles. In this case the rectangles are defined at the same points as where
    # we want to evaluate them. In the more general case, we would have to do 1D
    # interpolation, i.e., lower = np.interp(x_out, x, dyp[..., 0]).
    lower = np.copy(dyp[..., 0])
    upper = np.copy(dyp[..., 1])

    for j in range(len(x)):
        inside = (dxp[:, 0] <= x[j]) & (x[j] <= dxp[:, 1])
        lower_rect = np.min(dyp[inside, 0], initial=lower[j])
        upper_rect = np.max(dyp[inside, 1], initial=upper[j])
        lower[j] = min(lower[j], lower_rect)
        upper[j] = max(upper[j], upper_rect)

    ci = np.stack([lower, upper], axis=-1)
    return ci
