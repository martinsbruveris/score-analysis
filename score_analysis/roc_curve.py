"""
This module contains ROC curve calculations, including confidence bands for ROC curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .scores import DEFAULT_BOOTSTRAP_CONFIG, BinaryLabel, BootstrapConfig, Scores

ROC_CI_EXTRA_POINTS = 20  # Number of points added for ROC confidence band computations


@dataclass
class ROCCurve:
    """
    Class to hold the values and thresholds for an ROC curve, optionally with
    confidence bands.

    Args:
        fnr: False Negative Rate.
        fpr: False Positive Rate.
        thresholds: Thresholds.
        fnr_ci: (N, 2) array with FNR confidence bands.
        fpr_ci: (N, 2) array with FPR confidence bands.
    """

    fnr: np.ndarray
    fpr: np.ndarray
    thresholds: np.ndarray
    fnr_ci: Optional[np.ndarray] = None
    fpr_ci: Optional[np.ndarray] = None

    @property
    def tpr(self):
        """True Positive Rate"""
        return 1.0 - self.fnr

    @property
    def tnr(self):
        """True Negative Rate"""
        return 1.0 - self.fpr

    @property
    def frr(self):
        """False Rejection Rate (alias for FNR)"""
        return self.fnr

    @property
    def far(self):
        """False Acceptance Rate (alias for FPR)"""
        return self.fpr

    @property
    def tar(self):
        """True Acceptance Rate (alias for TPR)"""
        return self.tpr

    @property
    def trr(self):
        """True Rejection Rate (alias for TNR)"""
        return self.tnr

    @property
    def tpr_ci(self):
        """TPR CIs"""
        return None if self.fnr_ci is None else np.copy(1.0 - self.fnr_ci[..., ::-1])

    @property
    def tnr_ci(self):
        """TNR CIs"""
        return None if self.fpr_ci is None else np.copy(1.0 - self.fpr_ci[..., ::-1])

    @property
    def frr_ci(self):
        """FRR CIs (alias for FNR CIs)"""
        return self.fnr_ci

    @property
    def far_ci(self):
        """FAR CIs (alias for FPR CIs)"""
        return self.fpr_ci

    @property
    def tar_ci(self):
        """TAR CIs (Alias for FPR CIs)"""
        return self.tpr_ci

    @property
    def trr_ci(self):
        """TRR CIs (alias for TNR CIs)"""
        return self.tnr_ci


def roc(
    scores: Scores,
    *,
    fnr: Optional[ArrayLike] = None,
    fpr: Optional[ArrayLike] = None,
    thresholds: Optional[ArrayLike] = None,
    nb_points: Optional[int] = 100,
    x_axis: str = "fpr",
) -> ROCCurve:
    """
    Computes the ROC curve at the given FNR, FPR or threshold values.

    We can provide the FNR, FPR and threshold values at which to compute the ROC
    curve. We will use the union of these points for the ROC curve.

    If neither FNR, FPR, nor threshold values are provided, if ``nb_points`` is set,
    we use linearly spaced points between the smallest positive and negative scores.

    If ``nb_points`` is None, we use *all* positive and negative scores for the
    ROC curve. This gives the highest accuracy, but can be compute-intensive if we
    are working with large amount of scores.

    Args:
        scores: Scores for which to compute ROC curve.
        fnr: FNR points at which to compute the ROC curve.
        fpr: FPR points at which to compute the ROC curve.
        thresholds: Thresholds at which to compute the ROC curve.
        nb_points: Number of linearly spaced points to use, if neither one of FNR, FPR,
            nor thresholds are provided. If None, we use all scores for the ROC curve.
        x_axis: The values for x-axis metric will be increasing.

    Returns:
        A ROCCurve object with points on the ROC curve and the corresponding thresholds.
    """
    thresholds = _find_support_thresholds(
        scores, fnr, fpr, thresholds, nb_points, None, x_axis
    )

    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

    return ROCCurve(fnr=fnr, fpr=fpr, thresholds=thresholds)


def roc_with_ci(
    scores: Scores,
    *,
    fnr: Optional[ArrayLike] = None,
    fpr: Optional[ArrayLike] = None,
    thresholds: Optional[ArrayLike] = None,
    nb_points: Optional[int] = None,
    x_axis: str = "fpr",
    alpha: float = 0.05,
    config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
) -> ROCCurve:
    """
    Function to compute the confidence band around an ROC curve.

    We can provide the FNR, FPR and threshold values at which to compute the ROC
    curve. We will use the union of these points for the ROC curve.

    If neither FNR, FPR, nor threshold values are provided, if ``nb_points`` is set,
    we use linearly spaced points between the smallest positive and negative scores.

    If ``nb_points`` is None, we use *all* positive and negative scores for the
    ROC curve. This gives the highest accuracy, but can be compute-intensive if we
    are working with large amount of scores.

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
        x_axis: The values for x-axis metric will be increasing.
        alpha: Significance level. In range (0, 1).
        config: Bootstrap config.

    Returns:
        ROCCurve object with point values for the ROC curve and the lower and upper
        bounds of the confidence band values for both metric values.
    """
    thresholds = _find_support_thresholds(
        scores,
        fnr,
        fpr,
        thresholds,
        nb_points,
        ROC_CI_EXTRA_POINTS,
        x_axis,
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

    return ROCCurve(
        fnr=fnr, fpr=fpr, thresholds=thresholds, fnr_ci=fnr_band, fpr_ci=fpr_band
    )


def _find_support_thresholds(
    scores: Scores,
    fnr: Optional[np.ndarray],
    fpr: Optional[np.ndarray],
    thresholds: Optional[np.ndarray],
    nb_points: Optional[int],
    nb_extra_points: Optional[int],
    x_axis: str,
) -> np.ndarray:
    """
    This function contains the logic for combining the user-provided FNR, FPR, and
    threshold values, together with nb_points to one list of thresholds at which
    we will evaluate the ROC curve.

    If one of FNR, FPR or thresholds is provided, we use the union of thresholds
    corresponding to these values. In that case nb_points is ignored.

    If FNR, FPR and thresholds are all None, we use thresholds corresponding to
    linearly spaced points along FNR and FPR axes. To be precise, we use nb_points // 2
    points along the FNR axis and the same along the FPR axis.
    """
    if nb_extra_points is not None:
        # We subtract 4 to make space for thresholds beyond the pos and neg score
        # limits; see np.nextafter calls below.
        nb_extra_fnr_points = (nb_extra_points - 4) // 2
        nb_extra_fpr_points = nb_extra_points - 4 - nb_extra_fnr_points
    else:
        nb_extra_fnr_points = 0
        nb_extra_fpr_points = 0

    if thresholds is None:
        thresholds = np.zeros(shape=(0,))
    if fnr is not None:
        fnr_thresholds = scores.threshold_at_fnr(fnr)
        thresholds = np.concatenate([thresholds, fnr_thresholds])
    if fpr is not None:
        fpr_thresholds = scores.threshold_at_fpr(fpr)
        thresholds = np.concatenate([thresholds, fpr_thresholds])
    if len(thresholds) == 0:
        if nb_points is None:
            # Use all available scores as thresholds
            thresholds = np.concatenate([scores.pos, scores.neg])
        else:
            nb_fnr_points = nb_points // 2
            nb_fpr_points = nb_points - nb_fnr_points
            default_fnr = np.linspace(0.0, 1.0, nb_fnr_points, endpoint=True)
            fnr_thresholds = scores.threshold_at_fnr(default_fnr)

            default_fpr = np.linspace(0.0, 1.0, nb_fpr_points, endpoint=True)
            fpr_thresholds = scores.threshold_at_fpr(default_fpr)
            thresholds = np.concatenate([fnr_thresholds, fpr_thresholds])

    if nb_extra_points is not None:
        thresholds = np.sort(thresholds)

        fnr_min = np.min(scores.fnr(thresholds[[0, -1]]))
        fnr_max = np.max(scores.fnr(thresholds[[0, -1]]))
        fnr_extra = _add_extra_points(fnr_min, fnr_max, nb_extra_fnr_points)
        fnr_thresholds = scores.threshold_at_fnr(fnr_extra)

        fpr_min = np.min(scores.fpr(thresholds[[0, -1]]))
        fpr_max = np.max(scores.fpr(thresholds[[0, -1]]))
        fpr_extra = _add_extra_points(fpr_min, fpr_max, nb_extra_fpr_points)
        fpr_thresholds = scores.threshold_at_fpr(fpr_extra)

        thresholds = np.concatenate([thresholds, fnr_thresholds, fpr_thresholds])
        thresholds = np.concatenate(
            [
                thresholds,
                # Add thresholds that are smaller and larger than all scores
                [np.nextafter(scores.pos[0], -np.inf)],
                [np.nextafter(scores.pos[-1], np.inf)],
                [np.nextafter(scores.neg[0], -np.inf)],
                [np.nextafter(scores.neg[-1], np.inf)],
            ]
        )

    thresholds = np.sort(thresholds)
    if x_axis not in {"fnr", "fpr", "tnr", "tpr", "far", "frr", "tar", "trr"}:
        raise ValueError(f"Unknown value for x_axis: {x_axis}.")
    if x_axis in {"fpr", "tpr", "far", "tar"}:  # Decreasing metrics if score_class=pos
        thresholds = thresholds[::-1]
    if scores.score_class == BinaryLabel.neg:
        thresholds = thresholds[::-1]

    return thresholds


def _add_extra_points(x_min, x_max, nb_points: int) -> np.ndarray:
    nb_before = nb_points // 2
    nb_after = nb_points - nb_before
    x = np.array([])
    if x_min > 0.0:
        print("WAA")
        x_before = np.linspace(0.0, x_min, num=nb_before, endpoint=False)
        x = np.concatenate([x_before, x])
    if x_max < 1.0:
        print("WOOO")
        # Thresholds will be sorted later. We use the reverse order to take advantage
        # of endpoint=False to exclude x[-1].
        x_after = np.linspace(1.0, x_max, num=nb_after, endpoint=False)
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
