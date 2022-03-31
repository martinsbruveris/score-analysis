from __future__ import annotations

from enum import Enum
from typing import Any, Tuple, Union

import numpy as np

from .cm import BinaryConfusionMatrix


class BinaryLabel(Enum):
    pos = "pos"
    neg = "neg"


class Scores:
    def __init__(
        self,
        pos,
        neg,
        *,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
    ):
        """
        Args:
            pos: Scores for positive samples
            neg: Scores for negative samples
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?
        """
        self.pos = np.asarray(pos)
        self.neg = np.asarray(neg)
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        self.pos = np.sort(self.pos)
        self.neg = np.sort(self.neg)

    @staticmethod
    def from_labels(
        scores,
        labels,
        *,
        pos_label: Any = 1,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
    ) -> Scores:
        """
        Args:
            scores: Array with sample scores
            labels: Array with sample labels
            pos_label: The label of the positive class. All other labels are treated as
                negative labels.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?

        Returns:
            A Scores instance.
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        pos = scores[labels == pos_label]
        neg = scores[labels != pos_label]
        return Scores(pos, neg, score_class=score_class, equal_class=equal_class)

    def cm(self, threshold) -> BinaryConfusionMatrix:
        """
        Computes confusion matrices at the given threshold(s).

        Args:
            threshold: Can be a scalar or array-like.

        Returns:
            A BinaryConfusionMatrix.
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

        matrix = np.empty((*threshold.shape, 2, 2), dtype=int)
        matrix[..., 0, 0] = tp
        matrix[..., 0, 1] = fn
        matrix[..., 1, 0] = fp
        matrix[..., 1, 1] = tn

        return BinaryConfusionMatrix(matrix=matrix)

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

    def threshold_at_tpr(self, tpr):
        """Set threshold at True Positive Rate."""
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at TPR with no positive values.")
        return self._threshold_at_ratio(self.pos, tpr, False, BinaryLabel.pos)

    def threshold_at_fnr(self, fnr):
        """Set threshold at False Negative Rate."""
        if len(self.pos) == 0:
            raise ValueError("Cannot set threshold at FNR with no positive values.")
        return self._threshold_at_ratio(self.pos, fnr, True, BinaryLabel.pos)

    def threshold_at_tnr(self, tnr):
        """Set threshold at True Negative Rate."""
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at TNR with no negative values.")
        return self._threshold_at_ratio(self.neg, tnr, True, BinaryLabel.neg)

    def threshold_at_fpr(self, fpr):
        """Set threshold at False Positive Rate."""
        if len(self.neg) == 0:
            raise ValueError("Cannot set threshold at FPR with no negative values.")
        return self._threshold_at_ratio(self.neg, fpr, False, BinaryLabel.neg)

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

    def eer(self) -> Tuple[float, float]:
        """
        Calculates Equal Error Rate, i.e., where FPR = FNR.

        Returns:
            Tuple (threshold, eer) consisting of the threshold at which EER is achieved
            and the EER value.
        """
        sign = -(self.threshold_at_fpr(0.0) - self.threshold_at_fnr(0.0))

        # We consider the inverse functions, i.e., the function that map fpr/fnr to
        # the threshold and find the cross-over point using the bisection method.
        def f(x):
            y = self.threshold_at_fpr(x) - self.threshold_at_fnr(x)
            y = sign * y  # Normalize function, such that f(0) <= 0.
            return y

        # The function f is increasing, but not strictly, i.e., it can have flat spots,
        # so we find the left-most root, the right-most root and then take the average.
        left = self._find_root(f, 0.0, 1.0, find_first=True)
        right = self._find_root(f, 0.0, 1.0, find_first=False)

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
