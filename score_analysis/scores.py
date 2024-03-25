from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from . import utils
from .cm import ConfusionMatrix


class BinaryLabel(Enum):
    """
    Simple enum for positive and negative classes.
    """

    pos = "pos"
    neg = "neg"


SamplingMethod = Callable[["Scores"], "Scores"]

# TODO (martins): Fast bootstrap, say "single_pass". Calculate for each element, how
#   often it should be selected.
# TODO (martins): group2idx, so we can use integers for group indices. Should speed up
#   bootstrapping.
# TODO (martins): Add ids field, so we can quickly export ids with scores closes to
#   threshold. Idea is to make error-analysis more user-friendly.


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
    sampling_method: Union[str, SamplingMethod] = "replacement"
    stratified_sampling: Optional[str] = None
    smoothing: bool = False
    ratio: Optional[float] = None


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

    # Aliases for within Onfido use.
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

    def threshold_at_tpr(self, tpr):
        """Set threshold at True Positive Rate."""
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at TPR with no positive values.")
        # Example: We want threshold at 70% TPR. If easy_pos_ratio=60%, then we want
        # the threshold at 25% TPR on the remaining 40% hard positives, since
        # 70% - 60% = 10% is 25% of the remaining 40%
        tpr = np.maximum(np.asarray(tpr) - self.easy_pos_ratio, 0.0)
        tpr = np.minimum(tpr / self.hard_pos_ratio, 1.0)
        return self._threshold_at_ratio(self.pos, tpr, False, BinaryLabel.pos)

    def threshold_at_fnr(self, fnr):
        """Set threshold at False Negative Rate."""
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at FNR with no positive values.")
        # Example: We want the threshold at 5% FNR. If hard_pos_ratio=10%, then we want
        # the threshold at 5% / 0.1 = 50% of the available 10% of hard positives.
        fnr = np.minimum(np.asarray(fnr) / self.hard_pos_ratio, 1.0)
        return self._threshold_at_ratio(self.pos, fnr, True, BinaryLabel.pos)

    def threshold_at_tnr(self, tnr):
        """Set threshold at True Negative Rate."""
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at TNR with no negative values.")
        # See explanation in threshold_at_tpr()
        tnr = np.maximum(np.asarray(tnr) - self.easy_neg_ratio, 0.0)
        tnr = np.minimum(tnr / self.hard_neg_ratio, 1.0)
        return self._threshold_at_ratio(self.neg, tnr, True, BinaryLabel.neg)

    def threshold_at_fpr(self, fpr):
        """Set threshold at False Positive Rate."""
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at FPR with no negative values.")
        # See explanation at threshold_at_fnr()
        fpr = np.minimum(np.asarray(fpr) / self.hard_neg_ratio, 1.0)
        return self._threshold_at_ratio(self.neg, fpr, False, BinaryLabel.neg)

    def threshold_at_topr(self, topr):
        """
        Set threshold at Test Outcome Positive Rate.

        This is the proportion of samples where the test outcome is positive,
        i.e. the test detects the condition.
        """
        concat_scores = np.sort(np.concatenate([self.neg, self.pos]))
        if len(concat_scores) == 0:
            raise ValueError("Cannot set threshold at TOPR without any values.")
        # See explanation at threshold_at_tonr()
        easy_pos_to_total_ratio = self.nb_easy_pos / self.nb_all_samples
        topr = np.maximum(np.asarray(topr) - easy_pos_to_total_ratio, 0.0)
        topr = np.minimum(topr / self.hard_ratio, 1.0)
        return self._threshold_at_ratio(concat_scores, topr, False, BinaryLabel.pos)

    def threshold_at_tonr(self, tonr):
        """
        Set threshold at Test Outcome Negative Rate.

        This is the proportion of samples where the test outcome is negative,
        i.e. the test does not detect the condition.
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
        return self._threshold_at_ratio(concat_scores, tonr, True, BinaryLabel.neg)

    def _threshold_at_ratio(
        self, scores, target_ratio, increasing: bool, ratio_class: BinaryLabel
    ):
        """
        Helper function to set the threshold at a specific metric. The metrics covered
        are TPR, FPR, TNR and FNR.

        Parameters are with respect to the case
            score_class = "pos" and equal_class = "pos"
        The parameters have the following meaning
         - increasing states, whether the metric is increasing or decreasing
           with increasing scores.
         - ratio_class states, whether the metric is calculated from positives or
           negatives.
        The following table relates them to TPR, etc.

                 Increasing  ratio_class
            TPR     False        pos
            FNR     True         pos
            TNR     True         neg
            FPR     False        neg

        We compute a threshold, such that
            P(score <= threshold) = target_ratio, if right_continuous = True
            P(score < threshold) = target_ratio, if right_continuous = False
        """
        isscalar = np.isscalar(target_ratio)
        target_ratio = np.asarray(target_ratio)

        # The standard case is an increasing metric based on pos, i.e., FNR, with
        # equal_class and score_class equal to pos. This case is left-continuous.
        # Metrics based on neg, change the continuity behaviour at the sample points.
        left_continuous = ratio_class == BinaryLabel.pos
        # The equality class also determines the continuity
        if self.equal_class != BinaryLabel.pos:
            left_continuous = not left_continuous
        # We transform non-increasing metrics into increasing ones. Note that this does
        # not change continuity.
        if not increasing:
            target_ratio = 1.0 - target_ratio
        # And we transform the score direction as well
        if self.score_class != BinaryLabel.pos:
            target_ratio = 1.0 - target_ratio
            left_continuous = not left_continuous

        # From here on, we can pretend to be in the standard case, i.e., calculate
        # thresholds based on an increasing metric
        threshold = self._invert_increasing_function(
            scores, target_ratio, left_continuous
        )

        if isscalar:
            threshold = threshold.item()
        return threshold

    @staticmethod
    def _invert_increasing_function(scores, target_ratio, left_continuous: bool):
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

        threshold = la * scores[left_idx] + (1 - la) * scores[right_idx]
        threshold = np.asarray(threshold)  # We need this when target_ratio is scalar

        # Special cases of TPR <= 0. and TPR >= 1.
        threshold[target_ratio <= 0.0] = np.nextafter(scores[0], -np.inf)
        threshold[target_ratio >= 1.0] = np.nextafter(scores[-1], np.inf)

        return threshold

    # Aliases for within Onfido use.
    def threshold_at_tar(self, tar):
        """
        Set threshold at True Acceptance Rate

        Alias for :func:`~Scores.threshold_at_tpr`.
        """
        return self.threshold_at_tpr(tpr=tar)

    def threshold_at_frr(self, frr):
        """
        Set threshold at False Rejection Rate

        Alias for :func:`~Scores.threshold_at_fnr`.
        """
        return self.threshold_at_fnr(fnr=frr)

    def threshold_at_trr(self, trr):
        """
        Set threshold at True Rejection Rate

        Alias for :func:`~Scores.threshold_at_tnr`.
        """
        return self.threshold_at_tnr(tnr=trr)

    def threshold_at_far(self, far):
        """
        Set threshold at False Acceptance Rate

        Alias for :func:`~Scores.threshold_at_fpr`.
        """
        return self.threshold_at_fpr(fpr=far)

    def threshold_at_acceptance_rate(self, acceptance_rate):
        """
        Set threshold at Acceptance Rate

        Alias for :func:`~Scores.threshold_at_topr`.
        """
        return self.threshold_at_topr(topr=acceptance_rate)

    def threshold_at_rejection_rate(self, rejection_rate):
        """
        Set threshold at Rejection Rate

        Alias for :func:`~Scores.threshold_at_tonr`.
        """
        return self.threshold_at_tonr(tonr=rejection_rate)

    def threshold_at_metric(
        self,
        target,
        metric: Union[str, Callable],
        points: Optional[Union[int, np.ndarray]] = None,
    ):
        """
        General function for setting thresholds at arbitrary metrics. No assumption is
        made about the metric being monotone or the threshold being unique.

        Given a metric function and a target value, the function will find all values
        for the threshold such that metric(threshold) = target.

        If N = len(pos) + len(neg) is the number of scores and T = len(target) is the
        number of thresholds we want to set, this function has complexity O(N*T),
        because it searches over the whole score space to find all solutions. We can
        speed up the function by considering only a subset of points.

        Args:
            target: Target points at which to set the threshold.
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores, threshold: np.ndarray) -> np.ndarray
            points: If a scalar, we use this many linearly spaces scores between
                min(pos, neg) and max(pos, neg). If given an array, we evaluate the
                metric at exactly these points.

        Returns:
            A list threshold of the same length as target of arrays such that
            threshold[j] is a strictly increasing array containing all solutions of the
            equation metric(theta) = target[j].
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
            Tuple (threshold, eer) consisting of the threshold at which EER is achieved
            and the EER value.
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

    def _sample_indices(self, by_label: bool = False):
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
            p = (
                self.nb_all_pos / self.nb_all_samples
                if self.nb_all_samples > 0
                else 0.0
            )
            nb_pos = np.random.binomial(self.nb_all_samples, p)
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
        if config.sampling_method == "replacement":
            pos_idx, neg_idx, nb_easy_pos, nb_easy_neg = self._sample_indices(
                by_label=config.stratified_sampling == "by_label"
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
        elif config.sampling_method == "proportion":
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
        elif isinstance(config.sampling_method, str):
            raise ValueError(f"Unsupported sampling method {config.sampling_method}.")
        elif callable(config.sampling_method):
            scores = config.sampling_method(self)  # Custom sampling method
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
