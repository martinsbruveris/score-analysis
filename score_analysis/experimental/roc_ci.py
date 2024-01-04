from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats

from score_analysis import BinaryLabel, BootstrapConfig, Scores
from score_analysis.scores import DEFAULT_BOOTSTRAP_CONFIG
from score_analysis.tools import (
    _aggregate_rectangles,
    _apply_rule_of_three,
    _find_support_thresholds,
)
from score_analysis.utils import bootstrap_ci


@dataclass
class NormalDataset:
    mu_pos: float
    mu_neg: Optional[float] = None
    sigma_pos: float = 3.75
    sigma_neg: float = 3.0
    p_pos: float = 0.5
    n: Optional[int] = None
    score_class: Union[BinaryLabel, str] = "pos"

    def __post_init__(self):
        if self.mu_neg is None:
            self.mu_neg = -self.mu_pos

    @staticmethod
    def from_metrics(
        fnr: float,
        fpr: float,
        fnr_support: int,
        fpr_support: int,
        sigma_pos: float = 1.0,
        sigma_neg: float = 1.0,
    ) -> "NormalDataset":
        """
        Generates a dataset with normally distributed scores, such that at
        the threshold 0.0 we obtain the given FNR at FPR and there are
        fnr_support and fpr_support many false negatives and false positives
        respectively.
        """

        # We want P(X < 0) = fnr, where X ~ N(mu, si). We rewrite this as
        #     P((X - mu) / si < -mu / si) = fnr
        # and (X - mu) / si ~ N(0, 1), so -mu / si can be obtained as quantiles
        # of N(0, 1).
        mu_pos = -scipy.stats.norm.ppf(fnr) * sigma_pos
        nb_pos = int(fnr_support / fnr)

        # Similarly, we rewrite P(X < 0) = 1 - fpr as
        #     P((X - mu) / si < -mu / si) = 1 - fpr
        mu_neg = -scipy.stats.norm.ppf(1 - fpr) * sigma_neg
        nb_neg = int(fpr_support / fpr)

        n = nb_pos + nb_neg
        p_pos = nb_pos / n

        return NormalDataset(
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sigma_pos=sigma_pos,
            sigma_neg=sigma_neg,
            p_pos=p_pos,
            n=n,
            score_class="pos",
        )

    def sample(
        self, n: Optional[int] = None, *, p_pos: Optional[float] = None
    ) -> Scores:
        n = n if n is not None else self.n
        p_pos = p_pos if p_pos is not None else self.p_pos
        nb_pos = np.random.binomial(n, p_pos)
        nb_neg = n - nb_pos
        pos = np.random.normal(loc=self.mu_pos, scale=self.sigma_pos, size=nb_pos)
        neg = np.random.normal(loc=self.mu_neg, scale=self.sigma_neg, size=nb_neg)
        return Scores(pos=pos, neg=neg, score_class=self.score_class)

    def roc(
        self,
        *,
        fnr: Optional[np.ndarray] = None,
        fpr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if fnr is None and fpr is None:
            raise ValueError("Must provide either FNR or FPR.")
        if fnr is not None and fpr is not None:
            raise ValueError("Cannot provide both FNR and FPR.")

        if fnr is not None:
            threshold = scipy.stats.norm.ppf(fnr, loc=self.mu_pos, scale=self.sigma_pos)
        else:
            # isf is the inverse survival function (sf = 1 - cdf)
            threshold = scipy.stats.norm.isf(fpr, loc=self.mu_neg, scale=self.sigma_neg)

        fnr = scipy.stats.norm.cdf(threshold, loc=self.mu_pos, scale=self.sigma_pos)
        fpr = scipy.stats.norm.sf(threshold, loc=self.mu_neg, scale=self.sigma_neg)

        return fnr, fpr

    def threshold_at_fnr(self, fnr: np.ndarray) -> np.ndarray:
        threshold = scipy.stats.norm.ppf(fnr, loc=self.mu_pos, scale=self.sigma_pos)
        if np.isscalar(fnr):
            threshold = threshold.item()
        return threshold

    def threshold_at_fpr(self, fpr: np.ndarray) -> np.ndarray:
        # isf is the inverse survival function (sf = 1 - cdf)
        threshold = scipy.stats.norm.isf(fpr, loc=self.mu_neg, scale=self.sigma_neg)
        if np.isscalar(fpr):
            threshold = threshold.item()
        return threshold

    def fnr(self, threshold: np.ndarray) -> np.ndarray:
        fnr = scipy.stats.norm.cdf(threshold, loc=self.mu_pos, scale=self.sigma_pos)
        if np.isscalar(threshold):
            fnr = fnr.item()
        return fnr

    def fpr(self, threshold: np.ndarray) -> np.ndarray:
        fpr = scipy.stats.norm.sf(threshold, loc=self.mu_neg, scale=self.sigma_neg)
        if np.isscalar(threshold):
            fpr = fpr.item()
        return fpr


def fixed_width_band_ci(
    scores: Scores,
    *,
    fnr: Optional[np.ndarray] = None,
    fpr: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    nb_points: Optional[int] = None,
    alpha: float = 0.05,
    config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
):
    """
    Fixed-Width Bands (FWB) method to estimate CI bands from Campbell 1994. This was
    one of the benchmark methods in Macskassy et al. 2005.

    References:
        G. C. Campbell. Advances in statistical methodology for the evaluation of
            diagnostic and laboratory tests. Statistics in Medicine, vol 13, 499-508,
            1994.
        S. A. Macskassy, F. Provost and S. Rosset. ROC confidence bands: an empirical
            evaluation. ICML, 2005.

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
        (fnr, fpr, fnr_ci, fpr_ci): The point values for the ROC curve and the lower
        and upper bounds of the confidence band values for both metric values.
    """
    thresholds = _find_support_thresholds(
        scores=scores, fnr=fnr, fpr=fpr, thresholds=thresholds, nb_points=nb_points
    )
    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

    delta_samples = np.empty(config.nb_samples)
    # We use the inverse of the slope defined in Macskassy, since we have flipped the
    # axes. For us, FNR is on the x-axis.
    k = np.sqrt(len(scores.neg) / len(scores.pos))
    for j in range(config.nb_samples):
        sample = scores.bootstrap_sample(config=config)
        fnr_sample = sample.fnr(thresholds)
        fpr_sample = sample.fpr(thresholds)
        delta_samples[j] = _find_tube_radius(fnr, fpr, fnr_sample, fpr_sample, k)

    # We double alpha, because here we want the one-sided CI [0, upper] to have
    # coverage alpha.
    delta = bootstrap_ci(theta=delta_samples, alpha=2 * alpha, method="quantile")[1]

    v = np.array([delta, delta * k])
    fnr_plus, fpr_plus = _displace_curve(fnr, fpr, v)
    fnr_minus, fpr_minus = _displace_curve(fnr, fpr, -v)

    fnr_ci = [
        np.interp(fpr, fpr_minus[::-1], fnr_minus[::-1]),
        np.interp(fpr, fpr_plus[::-1], fnr_plus[::-1]),
    ]
    fnr_ci = np.stack(fnr_ci, axis=-1)
    fpr_ci = [np.interp(fnr, fnr_minus, fpr_minus), np.interp(fnr, fnr_plus, fpr_plus)]
    fpr_ci = np.stack(fpr_ci, axis=-1)

    return fnr, fpr, fnr_ci, fpr_ci


def _displace_curve(
    x: np.ndarray, y: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Displaces curve (x, y) by vector v while maintaining clipping to unit square."""
    x = np.clip(x + v[0], 0.0, 1.0)
    y = np.clip(y + v[1], 0.0, 1.0)

    # This is not general, but is sufficient for ROC curves: after displacement, we
    # still want the curve to go from (0, 1) to (1, 0), so we manually reset the start
    # and end points. A more general solution would look, when the curve is leaving the
    # unit square, etc.
    # We don't use exactly 1.0, but something slightly bigger, so the curve has a slope
    # and interpolation does not run into problems.
    x[0], y[0] = 0, np.nextafter(1.0, np.inf)
    x[-1], y[-1] = np.nextafter(1.0, np.inf), 0

    return x, y


def _find_tube_radius(
    x: np.ndarray, y: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: float
) -> float:
    """
    Finds the radius of the tube around the curve (x, y) that, when displaced along the
    slope k will contain the curve (xs, ys).

    All curves are assumed to start at (0, 1) and end at (1, 0) and are clipped to the
    unit square (0, 1)^2 during displacement.
    """
    v = np.array([1, k])

    def _is_contained(_delta):
        xp, yp = _displace_curve(x, y, _delta * v)
        yp = np.interp(xs, xp, yp)
        above = np.all(yp >= ys)

        xm, ym = _displace_curve(x, y, -_delta * v)
        ym = np.interp(xs, xm, ym)
        below = np.all(ym <= ys)
        return above and below

    if _is_contained(0.0):
        return 0.0
    # Radius 1.0 should be sufficient, but let's give us some margin of error here.
    # This part is a bit flaky and errors out a few times at mu_pos=3.0 and n=25, 50.
    if not _is_contained(4.0):
        raise ValueError("Could not initialise search for displacement.")

    tol = 1e-2
    delta_min, delta_max = 0.0, 1.0
    while delta_max - delta_min > tol:
        delta = (delta_max + delta_min) / 2
        if _is_contained(delta):
            delta_max = delta
        else:
            delta_min = delta
    return (delta_max + delta_min) / 2


def simultaneous_joint_region_ci(
    scores: Scores,
    *,
    fnr: Optional[np.ndarray] = None,
    fpr: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    nb_points: Optional[int] = None,
    alpha: float = 0.05,
    config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
):
    """
    Simultaneous Joint Confidence Region (SJR) method to estimate CI bands from
    Campbell 1994. This was one of the benchmark methods in Macskassy et al. 2005.

    References:
        G. C. Campbell. Advances in statistical methodology for the evaluation of
            diagnostic and laboratory tests. Statistics in Medicine, vol 13, 499-508,
            1994.
        S. A. Macskassy, F. Provost and S. Rosset. ROC confidence bands: an empirical
            evaluation. ICML, 2005.

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
        (fnr, fpr, fnr_ci, fpr_ci): The point values for the ROC curve and the lower
        and upper bounds of the confidence band values for both metric values.
    """
    thresholds = _find_support_thresholds(
        scores=scores, fnr=fnr, fpr=fpr, thresholds=thresholds, nb_points=nb_points
    )
    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

    fnr_delta = scipy.stats.ksone.ppf(1.0 - alpha / 2.0, scores.nb_all_pos)
    fpr_delta = scipy.stats.ksone.ppf(1.0 - alpha / 2.0, scores.nb_all_neg)

    fnr_ci = np.stack([fnr - fnr_delta, fnr + fnr_delta], axis=-1)
    fpr_ci = np.stack([fpr - fpr_delta, fpr + fpr_delta], axis=-1)

    # Here the magic happens, and we aggregate 1D CIs to a 2D confidence band.
    fpr_band = _aggregate_rectangles(fnr, fnr_ci, fpr_ci)
    fnr_band = _aggregate_rectangles(fpr, fpr_ci, fnr_ci)

    return fnr, fpr, fnr_band, fpr_band


def pointwise_band_ci(
    scores: Scores,
    *,
    fnr: Optional[np.ndarray] = None,
    fpr: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    nb_points: Optional[int] = None,
    alpha: float = 0.05,
    config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
):
    """
    Method that aggregates pointwise CI intervals into a confidence band. This is
    essentially the same as ``roc_with_ci``, except that we don't do aggregation of
    rectangles at the end.

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
        (fnr, fpr, fnr_ci, fpr_ci): The point values for the ROC curve and the lower
        and upper bounds of the confidence band values for both metric values.
    """
    thresholds = _find_support_thresholds(
        scores=scores, fnr=fnr, fpr=fpr, thresholds=thresholds, nb_points=nb_points
    )
    fnr = scores.fnr(thresholds)
    fpr = scores.fpr(thresholds)

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

    return fnr, fpr, fnr_ci, fpr_ci
