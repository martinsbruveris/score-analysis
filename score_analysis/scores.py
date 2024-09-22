from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from . import utils
from .cm import ConfusionMatrix


class BinaryLabel(Enum):
    """
    Simple enum for positive and negative classes.
    """

    pos = "pos"
    neg = "neg"


SAMPLING_METHOD_REPLACEMENT = "replacement"
SAMPLING_METHOD_SINGLE_PASS = "single_pass"
SAMPLING_METHOD_DYNAMIC = "dynamic"
SAMPLING_METHOD_PROPORTION = "proportion"

CustomSamplingMethod = Callable[["Scores"], "Scores"]
SamplingMethod = Union[str, CustomSamplingMethod]


@dataclass(frozen=True)
class BootstrapConfig:
    """
    Bootstrap configuration for creating bootstrap samples and computing CIs.

    Args:
        nb_samples: Number of samples to use for bootstrapping.
        bootstrap_method: Method to compute the CI from the bootstrap samples.
            Possible values are

             * "quantile" uses the alpha/2 and 1-alpha/2 quantiles of the
               empirical metric distribution.
             * "bc" applies bias correction to correct for the bias of the median
               of the empirical distribution
             * "bca" applies bias correction and acceleration to correct for non-
               constant standard error.

            See Ch. 11 of Computer Age Statistical Inference by Efron and Hastie
            for details.
        sampling_method: Sampling method to create bootstrap sample. Supported methods
            are

             * "replacement" creates a sample with the same number of positive and
               negative scores using sampling with replacement.
             * "single_pass" approximates replacement sampling using a Poisson
                distribution to determine how often each score would be selected in
                the bootstrap sample. This speeds up sampling by up to ~40% compared to
                replacement sampling since the sampled scores are already sorted.
                However, the method cannot guarantee that each bootstrap sample has the
                same number of scores. This should not matter if the number of scores
                is >100 (per group, if stratified sampling is used).
             * "dynamic" chooses between replacement and single pass sampling. It
                chooses single pass sampling, if >100 scores are present per group and
                smoothing is disabled and reverts to replacement sampling otherwise.
                The threshold can be changed by setting the variable
                ``SINGLE_PASS_SAMPLE_THRESHOLD``.
             * "proportion" creates a sample of size defined by ratio using sampling
               without replacement. This is similar to cross-validation, where a
               proportion of data is used in each iteration.
             * A callable with signature::

                  method(source: Scores, **kwargs) -> Scores

               creating one sample from a source Scores object.

        stratified_sampling: Stratified sampling is only supported for replacement
            sampling. Possible values are

             * ``None``. No stratification is used
             * "by_label". Sampling preserves the proportion of positive and negative
               samples as well as the proportion of easy positive and negative samples.
             * "by_group". Sampling preserves the proportion of samples in each group.
               Defaults to non-stratified sampling, if no groups are present.

        smoothing: Optional smoothing of sampled scores.
        ratio: Size of sample when using proportional sampling. In range (0, 1).
    """

    nb_samples: int = 1_000
    bootstrap_method: str = "bca"
    sampling_method: SamplingMethod = SAMPLING_METHOD_DYNAMIC
    stratified_sampling: Optional[str] = None
    smoothing: bool = False
    ratio: Optional[float] = None


SINGLE_PASS_SAMPLE_THRESHOLD = 100
DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()


class Scores:
    def __init__(
        self,
        pos,
        neg,
        *,
        nb_easy_pos: int = 0,
        nb_easy_neg: int = 0,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
        is_sorted: bool = False,
    ):
        """
        Args:
            pos: Scores for positive samples.
            neg: Scores for negative samples.
            nb_easy_pos: Number of positive samples that we assume are always correctly
                classified when computing metrics. These parameters when evaluating
                a highly accurate classifier on only the hardest samples to speed up
                evaluation.
            nb_easy_neg: Number of negative samples that we assume are always correctly
                classified.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?
            is_sorted: If True, we assume the scores are already sorted. Can be used to
                speed up Scores object creation.
        """
        self.pos = np.asarray(pos)
        self.neg = np.asarray(neg)
        self.nb_easy_pos = nb_easy_pos
        self.nb_easy_neg = nb_easy_neg
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        if not is_sorted:
            self.pos = np.sort(self.pos)
            self.neg = np.sort(self.neg)

    @property
    def hard_pos_ratio(self) -> float:
        if self.nb_easy_pos > 0:
            return len(self.pos) / (len(self.pos) + self.nb_easy_pos)
        else:
            # The default state is that all samples are hard
            return 1.0

    @property
    def hard_neg_ratio(self) -> float:
        if self.nb_easy_neg > 0:
            return len(self.neg) / (len(self.neg) + self.nb_easy_neg)
        else:
            # The default state is that all samples are hard
            return 1.0

    @property
    def easy_pos_ratio(self) -> float:
        return 1.0 - self.hard_pos_ratio

    @property
    def easy_neg_ratio(self) -> float:
        return 1.0 - self.hard_neg_ratio

    @property
    def nb_easy_samples(self) -> int:
        return self.nb_easy_pos + self.nb_easy_neg

    @property
    def nb_hard_pos(self) -> int:
        return len(self.pos)

    @property
    def nb_hard_neg(self) -> int:
        return len(self.neg)

    @property
    def nb_hard_samples(self) -> int:
        return len(self.pos) + len(self.neg)

    @property
    def nb_all_pos(self) -> int:
        return self.nb_easy_pos + len(self.pos)

    @property
    def nb_all_neg(self) -> int:
        return self.nb_easy_neg + len(self.neg)

    @property
    def nb_all_samples(self) -> int:
        return self.nb_easy_samples + self.nb_hard_samples

    @property
    def easy_ratio(self) -> float:
        if self.nb_easy_samples > 0:
            return self.nb_easy_samples / self.nb_all_samples
        else:
            # The default state is that all samples are hard
            return 0.0

    @property
    def hard_ratio(self) -> float:
        return 1.0 - self.easy_ratio

    @staticmethod
    def from_labels(
        labels,
        scores,
        *,
        pos_label: Any = 1,
        nb_easy_pos: int = 0,
        nb_easy_neg: int = 0,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
        is_sorted: bool = False,
    ) -> Scores:
        """
        Args:
            labels: Array with sample labels.
            scores: Array with sample scores.
            pos_label: The label of the positive class. All other labels are treated as
                negative labels.
            nb_easy_pos: Number of positive samples that we assume are always correctly
                classified when computing metrics. These parameters when evaluating
                a highly accurate classifier on only the hardest samples to speed up
                evaluation.
            nb_easy_neg: Number of negative samples that we assume are always correctly
                classified.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?
            is_sorted: If True, we assume the scores are already sorted. Can be used to
                speed up Scores object creation.

        Returns:
            A Scores instance.
        """
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        pos = scores[labels == pos_label]
        neg = scores[labels != pos_label]
        return Scores(
            pos=pos,
            neg=neg,
            nb_easy_pos=nb_easy_pos,
            nb_easy_neg=nb_easy_neg,
            score_class=score_class,
            equal_class=equal_class,
            is_sorted=is_sorted,
        )

    def __eq__(self, other: Scores) -> bool:
        """
        Tests if two Scores objects are equal. Equality is tested exactly, rounding
        errors can lead to objects being not equal.

        Args:
            other: Object to test against.

        Returns:
            True, if objects are equal, false otherwise.
        """
        equal = (
            np.array_equal(self.pos, other.pos)
            and np.array_equal(self.neg, other.neg)
            and self.nb_easy_pos == other.nb_easy_pos
            and self.nb_easy_neg == other.nb_easy_neg
            and self.score_class == other.score_class
            and self.equal_class == other.equal_class
        )
        return equal

    def swap(self) -> Scores:
        """
        Swaps positive and negative scores. Also reverses the decision logic, so that
        fpr of original scores equals fnr of reversed scores.

        Returns:
            Scores object with positive and negative scores reversed.
        """
        return Scores(
            pos=self.neg,
            neg=self.pos,
            nb_easy_pos=self.nb_easy_neg,
            nb_easy_neg=self.nb_easy_pos,
            score_class="neg" if self.score_class == BinaryLabel.pos else "pos",
            equal_class="neg" if self.equal_class == BinaryLabel.pos else "pos",
            is_sorted=True,
        )

    def cm(self, threshold) -> ConfusionMatrix:
        """
        Computes confusion matrices at the given threshold(s).

        Args:
            threshold: Can be a scalar or array-like.

        Returns:
            A binary confusion matrix.
        """
        threshold = np.asarray(threshold)

        if self.score_class == BinaryLabel.pos:
            side = "left" if self.equal_class == BinaryLabel.pos else "right"
        else:
            side = "right" if self.equal_class == BinaryLabel.pos else "left"

        pos_below = np.searchsorted(self.pos, threshold, side=side)
        neg_below = np.searchsorted(self.neg, threshold, side=side)

        pos_above = len(self.pos) - pos_below
        neg_above = len(self.neg) - neg_below

        if self.score_class == BinaryLabel.pos:
            tp, fn = pos_above, pos_below
            fp, tn = neg_above, neg_below
        else:
            tp, fn = pos_below, pos_above
            fp, tn = neg_below, neg_above

        # Account for easy samples
        tp += self.nb_easy_pos
        tn += self.nb_easy_neg

        matrix = np.empty((*threshold.shape, 2, 2), dtype=int)
        matrix[..., 0, 0] = tp
        matrix[..., 0, 1] = fn
        matrix[..., 1, 0] = fp
        matrix[..., 1, 1] = tn

        return ConfusionMatrix(matrix=matrix, binary=True)

    confusion_matrix = cm

    def tpr(self, threshold):
        """True Positive Rate at threshold(s)."""
        return self.cm(threshold).tpr()

    def fnr(self, threshold):
        """False Negative Rate at threshold(s)."""
        return self.cm(threshold).fnr()

    def tnr(self, threshold):
        """True Negative Rate at threshold(s)."""
        return self.cm(threshold).tnr()

    def fpr(self, threshold):
        """False Positive Rate at threshold(s)."""
        return self.cm(threshold).fpr()

    def topr(self, threshold):
        """Test Outcome Positive Rate at threshold(s)."""
        return self.cm(threshold).topr()

    def tonr(self, threshold):
        """Test Outcome Negative Rate at threshold(s)."""
        return self.cm(threshold).tonr()

    # Aliases.
    def tar(self, threshold):
        """
        True Acceptance Rate at threshold(s).

        Alias for :func:`~Scores.tpr`.
        """
        return self.tpr(threshold)

    def frr(self, threshold):
        """
        False Rejection Rate at threshold(s).

        Alias for :func:`~Scores.fnr`.
        """
        return self.fnr(threshold)

    def trr(self, threshold):
        """
        True Rejection Rate at threshold(s).

        Alias for :func:`~Scores.tnr`.
        """
        return self.tnr(threshold)

    def far(self, threshold):
        """
        False Acceptance Rate at threshold(s).

        Alias for :func:`~Scores.fpr`.
        """
        return self.fpr(threshold)

    def acceptance_rate(self, threshold):
        """
        Acceptance Rate at threshold(s).

        Alias for :func:`~Scores.topr`.
        """
        return self.topr(threshold)

    def rejection_rate(self, threshold):
        """
        Rejection Rate at threshold(s).

        Alias for :func:`~Scores.tonr`.
        """
        return self.tonr(threshold)

    def threshold_at_tpr(self, tpr, *, method: str = "linear"):
        """
        Set threshold at True Positive Rate.

        Args:
            tpr: TPR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at TPR with no positive values.")
        # Example: We want threshold at 70% TPR. If easy_pos_ratio=60%, then we want
        # the threshold at 25% TPR on the remaining 40% hard positives, since
        # 70% - 60% = 10% is 25% of the remaining 40%
        tpr = np.maximum(np.asarray(tpr) - self.easy_pos_ratio, 0.0)
        tpr = np.minimum(tpr / self.hard_pos_ratio, 1.0)
        return self._threshold_at_ratio(self.pos, tpr, False, BinaryLabel.pos, method)

    def threshold_at_fnr(self, fnr, *, method: str = "linear"):
        """
        Set threshold at False Negative Rate.

        Args:
            fnr: FNR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at FNR with no positive values.")
        # Example: We want the threshold at 5% FNR. If hard_pos_ratio=10%, then we want
        # the threshold at 5% / 0.1 = 50% of the available 10% of hard positives.
        fnr = np.minimum(np.asarray(fnr) / self.hard_pos_ratio, 1.0)
        return self._threshold_at_ratio(self.pos, fnr, True, BinaryLabel.pos, method)

    def threshold_at_tnr(self, tnr, *, method: str = "linear"):
        """
        Set threshold at True Negative Rate.

        Args:
            tnr: TNR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at TNR with no negative values.")
        # See explanation in threshold_at_tpr()
        tnr = np.maximum(np.asarray(tnr) - self.easy_neg_ratio, 0.0)
        tnr = np.minimum(tnr / self.hard_neg_ratio, 1.0)
        return self._threshold_at_ratio(self.neg, tnr, True, BinaryLabel.neg, method)

    def threshold_at_fpr(self, fpr, *, method: str = "linear"):
        """
        Set threshold at False Positive Rate.

        Args:
            fpr: FPR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at FPR with no negative values.")
        # See explanation at threshold_at_fnr()
        fpr = np.minimum(np.asarray(fpr) / self.hard_neg_ratio, 1.0)
        return self._threshold_at_ratio(self.neg, fpr, False, BinaryLabel.neg, method)

    def threshold_at_topr(self, topr, *, method: str = "linear"):
        """
        Set threshold at Test Outcome Positive Rate.

        This is the proportion of samples where the test outcome is positive,
        i.e. the test detects the condition.

        Args:
            topr: TOPR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        concat_scores = np.sort(np.concatenate([self.neg, self.pos]))
        if len(concat_scores) == 0:
            raise ValueError("Cannot set threshold at TOPR without any values.")
        # See explanation at threshold_at_tonr()
        easy_pos_to_total_ratio = self.nb_easy_pos / self.nb_all_samples
        topr = np.maximum(np.asarray(topr) - easy_pos_to_total_ratio, 0.0)
        topr = np.minimum(topr / self.hard_ratio, 1.0)
        return self._threshold_at_ratio(
            concat_scores, topr, False, BinaryLabel.pos, method
        )

    def threshold_at_tonr(self, tonr, *, method: str = "linear"):
        """
        Set threshold at Test Outcome Negative Rate.

        This is the proportion of samples where the test outcome is negative,
        i.e. the test does not detect the condition.

        Args:
            tonr: TONR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        concat_scores = np.sort(np.concatenate([self.neg, self.pos]))
        if len(concat_scores) == 0:
            raise ValueError("Cannot set threshold at TONR without any values.")
        # Example: Imagine we want a threshold at 85% TONR and 80% of our data are
        # easy negatives and 10% of our data are easy positives. Then, we want the
        # threshold at 50% TONR on the 10% of data for which we have scores, since
        # 85% - 80% = 5% is 50% of the 10% data with scores (5% / 10%).
        easy_neg_to_total_ratio = self.nb_easy_neg / self.nb_all_samples
        tonr = np.maximum(np.asarray(tonr) - easy_neg_to_total_ratio, 0.0)
        tonr = np.minimum(tonr / self.hard_ratio, 1.0)
        return self._threshold_at_ratio(
            concat_scores, tonr, True, BinaryLabel.neg, method
        )

    def _threshold_at_ratio(
        self,
        scores,
        target_ratio,
        increasing: bool,
        ratio_class: BinaryLabel,
        method: str,
    ):
        """
        Helper function to set the threshold at a specific metric, for metrics that
        are defined as ratios, such as TPR, FPR, TNR and FNR.

        The following table relates the parameters ``increasing`` and ``ratio_class``
        to common metrics, e.g., TPR, etc.

                 increasing  ratio_class
            TPR     False        pos
            FNR     True         pos
            TNR     True         neg
            FPR     False        neg

        We compute a threshold, such that
            P(score < threshold) = target_ratio, if metric is left continuous
            P(score <= threshold) = target_ratio, if metric is right continuous

        Args:
            scores: Array of scores to threshold
            target_ratio: The threshold is set such that the target_ratio of scores
                will be above or below the threshold. This can be an array.
            increasing: States, if the metric is increasing or decreasing, when
                score_class = "pos". Note that this is a property of the metric.
            ratio_class: States, if the metric is calculated using positive or
                negative scores.
            method: Possible values are "linear", "lower", "higher".
        """
        if method not in {"lower", "higher", "linear"}:
            raise ValueError(f"Unknown interpolation method: {method}.")

        # Dictionary for reversing the interpolation method.
        reverse_method = {"lower": "higher", "higher": "lower", "linear": "linear"}

        isscalar = np.isscalar(target_ratio)
        target_ratio = np.asarray(target_ratio)

        # The standard case is an increasing metric based on positive scores, i.e., FNR,
        # with equal_class and score_class equal to pos. This case is left-continuous.
        # Metrics based on neg, reverse the continuity behaviour.
        left_continuous = ratio_class == BinaryLabel.pos
        # The equality class also determines the continuity
        if self.equal_class != BinaryLabel.pos:
            left_continuous = not left_continuous
        # We transform non-increasing metrics into increasing ones. Note that this does
        # not change continuity.
        if not increasing:
            target_ratio = 1.0 - target_ratio
            method = reverse_method[method]
        # And we transform the score direction as well. This does change continuity.
        if self.score_class != BinaryLabel.pos:
            target_ratio = 1.0 - target_ratio
            left_continuous = not left_continuous
            method = reverse_method[method]
        # From here on, we can pretend to be in the standard case, i.e., calculate
        # thresholds based on an increasing metric.
        threshold = self._invert_increasing_function(
            scores, target_ratio, left_continuous, method
        )

        if isscalar:
            threshold = threshold.item()
        return threshold

    @staticmethod
    def _invert_increasing_function(
        scores, target_ratio, left_continuous: bool, method: str
    ):
        scores = scores.astype(float)  # Otherwise we can get problems with nextafter

        if not left_continuous:
            min_ratio = 1.0 / len(scores)
            target_ratio = target_ratio - min_ratio
        target = target_ratio * len(scores)

        left_idx = np.floor(target)  # Element to left of threshold
        right_idx = np.ceil(target)  # Element to right of threshold
        la = right_idx - target  # Coefficient in convex sum

        # Ensure we don't exceed array limits
        left_idx = np.maximum(np.minimum(left_idx.astype(int), len(scores) - 1), 0)
        right_idx = np.maximum(np.minimum(right_idx.astype(int), len(scores) - 1), 0)

        if method == "linear":
            threshold = la * scores[left_idx] + (1 - la) * scores[right_idx]
        elif method == "lower":
            threshold = scores[left_idx]
        else:  # "higher"
            threshold = scores[right_idx]
        threshold = np.asarray(threshold)  # We need this when target_ratio is scalar

        # Special cases of TPR <= 0. and TPR >= 1.
        threshold[target_ratio <= 0.0] = np.nextafter(scores[0], -np.inf)
        threshold[target_ratio >= 1.0] = np.nextafter(scores[-1], np.inf)

        return threshold

    # Aliases.
    def threshold_at_tar(self, tar, *, method: str = "linear"):
        """
        Set threshold at True Acceptance Rate

        Alias for :func:`~Scores.threshold_at_tpr`.
        """
        return self.threshold_at_tpr(tpr=tar, method=method)

    def threshold_at_frr(self, frr, *, method: str = "linear"):
        """
        Set threshold at False Rejection Rate

        Alias for :func:`~Scores.threshold_at_fnr`.
        """
        return self.threshold_at_fnr(fnr=frr, method=method)

    def threshold_at_trr(self, trr, *, method: str = "linear"):
        """
        Set threshold at True Rejection Rate

        Alias for :func:`~Scores.threshold_at_tnr`.
        """
        return self.threshold_at_tnr(tnr=trr, method=method)

    def threshold_at_far(self, far, *, method: str = "linear"):
        """
        Set threshold at False Acceptance Rate

        Alias for :func:`~Scores.threshold_at_fpr`.
        """
        return self.threshold_at_fpr(fpr=far, method=method)

    def threshold_at_acceptance_rate(self, acceptance_rate, *, method: str = "linear"):
        """
        Set threshold at Acceptance Rate

        Alias for :func:`~Scores.threshold_at_topr`.
        """
        return self.threshold_at_topr(topr=acceptance_rate, method=method)

    def threshold_at_rejection_rate(self, rejection_rate, *, method: str = "linear"):
        """
        Set threshold at Rejection Rate

        Alias for :func:`~Scores.threshold_at_tonr`.
        """
        return self.threshold_at_tonr(tonr=rejection_rate, method=method)

    def threshold_at_metric(
        self,
        target,
        metric: Union[str, Callable],
        points: Optional[Union[int, np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        General function for setting thresholds at arbitrary metrics. No assumption is
        made about the metric being monotone or the threshold being unique.

        Given a metric function and a target value, the function will find all values
        for the threshold such that ``metric(threshold) = target``.

        If ``N = len(pos) + len(neg)`` is the number of scores and ``T = len(target)``
        is the number of thresholds we want to set, this function has complexity O(N*T),
        because it searches over the whole score space to find all solutions. We can
        speed up the function by considering only a subset of points.

        Args:
            target: Target points at which to set the threshold.
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores, threshold: np.ndarray) -> np.ndarray
            points: If a scalar, we use this many linearly spaced scores between
                ``min(pos, neg)`` and ``max(pos, neg)``. If given an array, we evaluate
                the metric at exactly these points.

        Returns:
            A list of thresholds of the same length as ``target``, such that
            ``threshold[j]`` is a strictly increasing array containing all solutions of
            the equation ``metric(theta) = target[j]``.
        """
        if isinstance(metric, str):
            metric = getattr(type(self), metric)

        if points is None:
            points = np.sort(np.concatenate([self.pos, self.neg]))
            if len(points) < 2:
                raise ValueError("At least two values are required to set thresholds.")
        elif isinstance(points, int):
            min_score = min(
                self.pos[0] if len(self.pos) > 0 else np.inf,
                self.neg[0] if len(self.neg) > 0 else np.inf,
            )
            max_score = max(
                self.pos[-1] if len(self.pos) > 0 else -np.inf,
                self.neg[-1] if len(self.neg) > 0 else -np.inf,
            )
            if min_score >= max_score:
                raise ValueError("At least two values are required to set thresholds.")
            points = np.linspace(min_score, max_score, points, endpoint=True)

        threshold = utils.invert_pl_function(x=points, y=metric(self, points), t=target)
        return threshold

    def eer(self) -> Tuple[float, float]:
        """
        Calculates Equal Error Rate, i.e., where FPR = FNR (or, equivalently, where
        FAR = FRR).

        Returns:
            Tuple ``(threshold, eer)``, consisting of the threshold at which EER is
            achieved and the EER value itself.
        """
        # We treat the case of perfect separation separately
        if self.pos[0] >= self.neg[-1] and self.score_class == BinaryLabel.pos:
            return (self.pos[0] + self.neg[-1]) / 2, 0.0
        if self.pos[-1] <= self.neg[0] and self.score_class == BinaryLabel.neg:
            return (self.pos[-1] + self.neg[0]) / 2, 0.0

        sign = -(self.threshold_at_fpr(0.0) - self.threshold_at_fnr(0.0))

        # We consider the inverse functions, i.e., the function that map fpr/fnr to
        # the threshold and find the cross-over point using the bisection method.
        def f(x):
            y = self.threshold_at_fpr(x) - self.threshold_at_fnr(x)
            y = sign * y  # Normalize function, such that f(0) <= 0.
            return y

        # EER cannot be larger than this: FPR cannot be larger than hard_pos_ratio,
        # since all other positives are easy and thus always correctly classified;
        # similarly, FNR cannot be larger than hard_neg_ratio
        max_eer = min(self.hard_pos_ratio, self.hard_neg_ratio)

        # This is the case of a very bad classifier, i.e., without easy samples this
        # happens only if all samples are incorrectly classified, i.e., EER=1.0. With
        # easy samples, we have to interpolate at the edge of where hard samples stop.
        if f(max_eer) < 0:
            if np.isclose(self.hard_pos_ratio, self.hard_neg_ratio):
                threshold = (
                    self.threshold_at_fpr(max_eer) + self.threshold_at_fnr(max_eer)
                ) / 2
                return threshold, max_eer
            elif self.hard_pos_ratio < self.hard_neg_ratio:
                threshold = self.threshold_at_fpr(self.hard_pos_ratio)
                return threshold, self.hard_pos_ratio
            else:
                threshold = self.threshold_at_fnr(self.hard_neg_ratio)
                return threshold, self.hard_neg_ratio

        # The function f is increasing, but not strictly, i.e., it can have flat spots,
        # so we find the left-most root, the right-most root and then take the average.
        left = self._find_root(f, 0.0, max_eer, find_first=True)
        right = self._find_root(f, 0.0, max_eer, find_first=False)

        eer = (left + right) / 2
        threshold = self.threshold_at_fpr(eer)

        return threshold, eer

    def auc(
        self,
        lower: float = 0.0,
        upper: float = 1.0,
        *,
        x_axis: str = "fpr",
        y_axis: str = "tpr",
    ):
        """
        Computes the (partial) AUC for the given Scores object using the trapezoid
        integration rule.

        Args:
            lower: Lower limit of integration.
            upper: Upper limit of integration.
            x_axis: Metric to plot on x-axis. Defaults to FPR.
            y_axis: Metric to plot on y-axis. Defaults to TPR.
        """
        # We add +/-eps to each score to deal with continuity properties of the ROC
        # curve. AUC is invariant to left-/right-continuity, but the metrics have
        # varying continuity behaviour at the score values. Duplicating the scores
        # makes the function slower, but it means that we don't have to figure out
        # continuity properties by hand.
        points = np.nextafter(
            np.concatenate([self.pos, self.neg]), [[-np.inf], [np.inf]]
        )
        points = np.sort(points.flatten())

        x = getattr(self, x_axis)(points)
        y = getattr(self, y_axis)(points)

        if x[-1] < x[0]:
            x = x[::-1]
            y = y[::-1]
        left = np.searchsorted(x, lower, side="left")  # x[l - 1] < lower <= x[l]
        right = np.searchsorted(x, upper, side="right")  # x[r - 1] <= upper < x[r]

        # Ensure indices are in range
        left = np.minimum(left, len(y) - 1)
        right = np.maximum(right, 1)

        x = np.concatenate([[lower], x[left:right], [upper]])
        y = np.concatenate([[y[left]], y[left:right], [y[right - 1]]])

        try:
            trapezoid = np.trapezoid  # Only available in Numpy 2.x
        except AttributeError:  # pragma: no cover
            trapezoid = np.trapz  # Deprecated
        # We use abs, so we don't have to worry about increasing/decreasing values.
        return np.abs(trapezoid(y, x))

    @staticmethod
    def _find_root(f, xa, xe, find_first, xtol=1e-10) -> float:
        """Finds first or last root of a monotone function on interval (xa, xe)."""
        if not (f(xa) <= 0 <= f(xe)):
            raise ValueError(f"f({xa}) <= 0 <= f({xe}) not satisfied.")

        while not np.abs(xa - xe) < xtol:
            xm = (xa + xe) / 2
            if f(xm) < 0:
                xa = xm
            elif f(xm) > 0:
                xe = xm
            else:
                if find_first:
                    # If we care about the first root, we move the end of
                    # the interval to the left.
                    xe = xm
                else:
                    xa = xm

        return (xa + xe) / 2

    def _sampling_method(self, config: BootstrapConfig) -> SamplingMethod:
        """
        If sampling method is "dynamic", we select the appropriate method between
        replacement and single pass, depending on the number of scores and whether
        smoothing is enabled.
        """
        if config.sampling_method != SAMPLING_METHOD_DYNAMIC:
            return config.sampling_method  # Nothing to choose here.

        if (
            self.nb_hard_pos < SINGLE_PASS_SAMPLE_THRESHOLD
            or self.nb_hard_neg < SINGLE_PASS_SAMPLE_THRESHOLD
            or config.smoothing
        ):
            return SAMPLING_METHOD_REPLACEMENT
        else:
            return SAMPLING_METHOD_SINGLE_PASS

    def _sample_indices(self, by_label: bool = False, single_pass: bool = False):
        """
        Returns the indices of positive and negative scores that we would include in
        the bootstrap sample.
        """
        if by_label:
            # Stratified sampling does not change easy : hard and pos : neg ratios.
            nb_easy_pos = self.nb_easy_pos
            nb_easy_neg = self.nb_easy_neg
            nb_hard_pos = self.nb_hard_pos
            nb_hard_neg = self.nb_hard_neg
        else:
            # Non-stratified sampling also takes into account uncertainty about the
            # pos : neg ratio in the bootstrapped sample.
            pos_neg_ratio = (
                self.nb_all_pos / self.nb_all_samples
                if self.nb_all_samples > 0
                else 0.0
            )
            nb_pos = np.random.binomial(self.nb_all_samples, pos_neg_ratio)
            nb_neg = self.nb_all_samples - nb_pos

            # Try to have at least one positive and one negative sample.
            if nb_pos == 0 and self.nb_all_pos > 0:
                nb_pos, nb_neg = 1, self.nb_all_samples - 1
            if nb_neg == 0 and self.nb_all_neg > 0:
                nb_pos, nb_neg = self.nb_all_samples - 1, 1

            # Find out how many easy and hard samples there will be.
            nb_easy_pos = np.random.binomial(nb_pos, self.easy_pos_ratio)
            nb_easy_neg = np.random.binomial(nb_neg, self.easy_neg_ratio)
            nb_hard_pos = nb_pos - nb_easy_pos
            nb_hard_neg = nb_neg - nb_easy_neg

            # Try to have at least one hard positive and negative sample.
            if nb_hard_pos == 0 and self.nb_hard_pos > 0:
                nb_hard_pos, nb_easy_pos = 1, nb_pos - 1
            if nb_hard_neg == 0 and self.nb_hard_neg > 0:
                nb_hard_neg, nb_easy_neg = 1, nb_neg - 1

        if single_pass:

            def _single_pass_sampling(size, n, p):
                # This threshold is toggling between using the binomial distribution
                # to choose how often each score will be included in the bootstrap
                # sample and approximating the binomial distribution with the Poisson
                # distribution. The Poisson distribution is a sufficiently good
                # approximation, if n > 100 and n*p < 20 (in our case, n*p is ~1).
                #
                # This is different from SINGLE_PASS_SAMPLE_THRESHOLD, which toggles
                # between single pass and replacement sampling. This is all single
                # pass sampling.
                if n < 100:
                    return np.random.binomial(size=size, n=n, p=p)
                else:
                    return np.random.poisson(size=size, lam=n * p)

            nb_pos_selected = _single_pass_sampling(
                size=self.nb_hard_pos, n=nb_hard_pos, p=1.0 / self.nb_hard_pos
            )
            nb_neg_selected = _single_pass_sampling(
                size=self.nb_hard_neg, n=nb_hard_neg, p=1.0 / self.nb_hard_neg
            )

            pos_idx = np.repeat(np.arange(self.nb_hard_pos), nb_pos_selected)
            neg_idx = np.repeat(np.arange(self.nb_hard_neg), nb_neg_selected)
        else:
            # Sampling hard samples with replacement
            pos_idx = np.random.choice(self.nb_hard_pos, size=nb_hard_pos, replace=True)
            neg_idx = np.random.choice(self.nb_hard_neg, size=nb_hard_neg, replace=True)

        return pos_idx, neg_idx, nb_easy_pos, nb_easy_neg

    def bootstrap_sample(
        self,
        config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
    ) -> Scores:
        """
        Creates one bootstrap sample by sampling with the specified configuration.

        Args:
            config: Bootstrap configuration.

        Returns:
            Scores object with the sampled scores.
        """
        # Resolve "dynamic" sampling to either replacement or single pass sampling.
        sampling_method = self._sampling_method(config)

        if sampling_method == SAMPLING_METHOD_REPLACEMENT:
            pos_idx, neg_idx, nb_easy_pos, nb_easy_neg = self._sample_indices(
                by_label=config.stratified_sampling == "by_label", single_pass=False
            )

            pos = self.pos[pos_idx]
            neg = self.neg[neg_idx]

            if config.smoothing:

                def _estimate_bandwidth(x: np.ndarray) -> float:
                    # The rule for the bandwidth estimation is taken from the wiki page
                    # https://en.wikipedia.org/wiki/Kernel_density_estimation
                    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
                    h = 0.9 * min(x.std(), iqr / 1.34) * math.pow(len(x), -0.2)
                    return h

                h_pos = _estimate_bandwidth(pos)
                h_neg = _estimate_bandwidth(neg)
                pos = pos + np.random.normal(loc=0.0, scale=h_pos, size=pos.shape)
                neg = neg + np.random.normal(loc=0.0, scale=h_neg, size=neg.shape)

            scores = Scores(
                pos=pos,
                neg=neg,
                nb_easy_pos=nb_easy_pos,
                nb_easy_neg=nb_easy_neg,
                score_class=self.score_class,
                equal_class=self.equal_class,
            )
        elif sampling_method == SAMPLING_METHOD_SINGLE_PASS:
            pos_idx, neg_idx, nb_easy_pos, nb_easy_neg = self._sample_indices(
                by_label=config.stratified_sampling == "by_label", single_pass=True
            )

            pos = self.pos[pos_idx]
            neg = self.neg[neg_idx]

            if config.smoothing:
                # We don't support smoothing, because after smoothing the scores would
                # not be sorted anymore, which would remove the main source of speedup
                # for the method.
                raise ValueError("Smoothing is not supported for single pass sampling.")

            scores = Scores(
                pos=pos,
                neg=neg,
                nb_easy_pos=nb_easy_pos,
                nb_easy_neg=nb_easy_neg,
                score_class=self.score_class,
                equal_class=self.equal_class,
                is_sorted=True,  # Single-pass sampling returns already sorted scores.
            )
        elif sampling_method == SAMPLING_METHOD_PROPORTION:
            if config.ratio is None:
                raise ValueError("For proportional sampling, ratio has to be defined.")
            # Sampling a sample defined by ratio, without replacement
            nb_pos = max(int(config.ratio * self.pos.size), 1)
            nb_neg = max(int(config.ratio * self.neg.size), 1)
            pos = np.random.choice(self.pos, size=nb_pos, replace=False)
            neg = np.random.choice(self.neg, size=nb_neg, replace=False)

            # We also "sample" a proportional sample of easy sample
            nb_easy_pos = int(config.ratio * self.nb_easy_pos)
            nb_easy_neg = int(config.ratio * self.nb_easy_neg)

            scores = Scores(
                pos=pos,
                neg=neg,
                nb_easy_pos=nb_easy_pos,
                nb_easy_neg=nb_easy_neg,
                score_class=self.score_class,
                equal_class=self.equal_class,
            )
        elif isinstance(sampling_method, str):
            raise ValueError(f"Unsupported sampling method {config.sampling_method}.")
        elif callable(sampling_method):
            scores = sampling_method(self)  # Custom sampling method
        else:
            raise ValueError("Sampling method must be a string or a callable.")

        return scores

    def bootstrap_metric(
        self,
        metric: Union[str, Callable],
        config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates nb_samples samples of metric using bootstrapping.

        Args:
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores, **kwargs) -> np.ndarray
            config: Bootstrap config.
            **kwargs: Arguments that are passed to the metric function.

        Returns:
            Array of samples from metric. If metric returns arrays of shape (X,), the
            function will return an array of shape (nb_samples, X).
        """
        if isinstance(metric, str):
            # getattr(self) would resolve the method from Scores, while type(self) will
            # return the method from the subclass, e.g., from GroupScores.
            metric = getattr(type(self), metric)

        m = np.asarray(metric(self, **kwargs))
        res = np.empty(shape=(config.nb_samples, *m.shape), dtype=m.dtype)
        for j in range(config.nb_samples):
            sample = self.bootstrap_sample(config=config)
            res[j] = metric(sample, **kwargs)

        return res

    def bootstrap_ci(
        self,
        metric: Union[str, Callable],
        alpha: float = 0.05,
        config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates the confidence interval with approximate coverage 1-alpha for metric
        by bootstrapping nb_samples from the positive and negative scores.

        Args:
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores, **kwargs) -> Union[float, np.ndarray]
            alpha: Significance level. In range (0, 1).
            config: Bootstrap config.
            **kwargs: Arguments that are passed to the metric function.

        Returns:
            Returns an array of shape (Y, 2) with lower and upper bounds of the CI, for
            a metric returning shape (Y,).
        """
        if isinstance(metric, str):
            metric = getattr(type(self), metric)
        samples = self.bootstrap_metric(metric, config=config, **kwargs)  # (N, Y)
        ci = utils.bootstrap_ci(
            theta=samples,
            theta_hat=metric(self, **kwargs),
            alpha=alpha,
            method=config.bootstrap_method,
        )
        return ci


def pointwise_cm(
    labels,
    scores,
    threshold,
    *,
    pos_label: Any = 1,
    score_class: Union[BinaryLabel, str] = "pos",
    equal_class: Union[BinaryLabel, str] = "pos",
) -> np.ndarray:
    """
    Returns a boolean array, stating for each score, in which field of the confusion
    matrix it belongs at a given threshold.

    If scores have shape (X,), threshold has shape (Y,), then we return an array of
    shape (X, Y, 2, 2).

    Args:
        labels: Array with sample labels
        scores: Array with sample scores
        threshold: Thresholds at which to classify scores
        pos_label: The label of the positive class. All other labels are treated as
            negative labels.
        score_class: Do scores indicate membership of the positive or the negative
            class?
        equal_class: Do samples with score equal to the threshold get assigned to
            the positive or negative class?

    Returns:
        Boolean array of shape (X, Y, 2, 2) specifying whether a given sample at a
        given threshold belongs in a given cell of the confusion matrix.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    threshold = np.asarray(threshold)
    score_class = BinaryLabel(score_class)
    equal_class = BinaryLabel(equal_class)

    # Save the shape and flatten objects
    scores_shape = scores.shape
    labels = np.reshape(labels, -1)
    labels = labels[:, np.newaxis]
    scores = np.reshape(scores, -1)
    scores = scores[:, np.newaxis]
    threshold_shape = threshold.shape
    threshold = np.reshape(threshold, -1)
    threshold = threshold[np.newaxis, :]

    # True labels
    pos = labels == pos_label
    neg = labels != pos_label

    # Predicted labels
    if score_class == BinaryLabel.pos and equal_class == BinaryLabel.pos:
        top = scores >= threshold  # Test Outcome Positive
        ton = scores < threshold  # Test Outcome Negative
    elif score_class == BinaryLabel.pos and equal_class == BinaryLabel.neg:
        top = scores > threshold
        ton = scores <= threshold
    elif score_class == BinaryLabel.neg and equal_class == BinaryLabel.pos:
        top = scores <= threshold
        ton = scores > threshold
    else:  # score_class == BinaryLabel.neg and equal_class == BinaryLabel.neg
        top = scores < threshold
        ton = scores >= threshold

    # The confusion matrix
    cm = np.empty((scores.size, threshold.size, 2, 2), dtype=bool)
    cm[..., 0, 0] = pos & top
    cm[..., 0, 1] = pos & ton
    cm[..., 1, 0] = neg & top
    cm[..., 1, 1] = neg & ton

    # Restore shapes
    cm = np.reshape(cm, (-1, *threshold_shape, 2, 2))
    cm = np.reshape(cm, (*scores_shape, *threshold_shape, 2, 2))

    return cm
