from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .cm import ConfusionMatrix


class BinaryLabel(Enum):
    """
    Simple enum for positive and negative classes.
    """

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
        labels,
        scores,
        *,
        pos_label: Any = 1,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
    ) -> Scores:
        """
        Args:
            labels: Array with sample labels
            scores: Array with sample scores
            pos_label: The label of the positive class. All other labels are treated as
                negative labels.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?

        Returns:
            A Scores instance.
        """
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        pos = scores[labels == pos_label]
        neg = scores[labels != pos_label]
        return Scores(pos, neg, score_class=score_class, equal_class=equal_class)

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
            and self.score_class == other.score_class
            and self.equal_class == other.equal_class
        )
        return equal

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
        # We treat the case of perfect separation separately
        if self.pos[0] >= self.neg[-1]:
            eer = 0.0 if self.score_class == BinaryLabel.pos else 1.0
            return (self.pos[0] + self.neg[-1]) / 2, eer
        if self.pos[-1] <= self.neg[0]:
            eer = 1.0 if self.score_class == BinaryLabel.pos else 0.0
            return (self.pos[-1] + self.neg[0]) / 2, eer

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

    def bootstrap_sample(
        self,
        method: Union[str, Callable] = "replacement",
        ratio: Optional[float] = None,
    ) -> Scores:
        """
        Creates one bootstrap sample by sampling with the specified method.

        Supported methods are

        * "replacement" creates a sample with the same number of positive and negative
          scores using sampling with replacement.
        * "proportion" creates a sample of size defined by ratio using sampling without
          replacement. This is similar to cross-validation, where a proportion of data
          is used in each iteration.
        * A callable with signature::

            method(source: Scores) -> Scores

          creating one sample from a source Scores object.

        Args:
            method: Sampling method to create bootstrap sample. One of "replacement" or
                "proportion".
            ratio: Size of sample when using proportional sampling. In range (0, 1).

        Returns:
            Scores object with the sampled scores.
        """
        if isinstance(method, str):
            if method == "replacement":
                # Sampling a sample of the same size with replacement
                pos = np.random.choice(self.pos, size=self.pos.size, replace=True)
                neg = np.random.choice(self.neg, size=self.neg.size, replace=True)
                # This code also takes into account uncertainty about the pos : neg
                # ratio in the bootstrapped sample. Not used at the moment.
                # n = self.pos.size + self.neg.size
                # p = self.pos.size / n
                # nb_pos = np.random.binomial(n, p)
                # pos = np.random.choice(self.pos, size=nb_pos, replace=True)
                # neg = np.random.choice(self.neg, size=n - nb_pos, replace=True)
            elif method == "proportion":
                if ratio is None:
                    raise ValueError(
                        "For proportional sampling, ratio has to be defined."
                    )
                # Sampling a sample defined by ratio, without replacement
                pos_size = max(int(ratio * self.pos.size), 1)
                pos = np.random.choice(self.pos, size=pos_size, replace=False)
                neg_size = max(int(ratio * self.neg.size), 1)
                neg = np.random.choice(self.neg, size=neg_size, replace=False)
            else:
                raise ValueError(f"Unsupported sampling method {method}.")

            scores = Scores(
                pos, neg, score_class=self.score_class, equal_class=self.equal_class
            )
        elif callable(method):
            scores = method(self)  # Custom sampling method
        else:
            raise ValueError("Method must be a string or a callable.")

        return scores

    def bootstrap_metric(
        self,
        metric: Union[str, Callable],
        *,
        nb_samples: int = 1000,
        method: Union[str, Callable] = "replacement",
        ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calculates nb_samples samples of metric using bootstrapping.

        Args:
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores) -> np.ndarray
            nb_samples: Number of samples to return
            method: Sampling method to create bootstrap sample. One of "replacement" or
                "proportion".
            ratio: Size of sample when using proportional sampling. In range (0, 1).

        Returns:
            Array of samples from metric. If metric returns arrays of shape (X,), the
            function will return an array of shape (nb_samples, X).
        """
        if isinstance(metric, str):
            metric = getattr(Scores, metric)

        m = np.asarray(metric(self))
        res = np.empty(shape=(nb_samples, *m.shape), dtype=m.dtype)
        for j in range(nb_samples):
            sample = self.bootstrap_sample(method=method, ratio=ratio)
            res[j] = metric(sample)

        return res

    def bootstrap_ci(
        self,
        metric: Union[str, Callable],
        alpha: float = 0.05,
        *,
        nb_samples: int = 1000,
        method: Union[str, Callable] = "replacement",
        ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calculates the confidence interval with approximate coverage 1-alpha for metric
        by bootstraping nb_samples from the positive and negative scores.

        Args:
            metric: Can be a string indicating a member function of the Scores class
                or a callable with signature::

                    metric(sample: Scores) -> Union[float, np.ndarray]
            alpha: Significance level. In range (0, 1).
            nb_samples: Number of samples to bootstrap
            method: Sampling method to create bootstrap sample. One of "replacement" or
                "proportion".
            ratio: Size of sample when using proportional sampling. In range (0, 1).

        Returns:
            Returns an array of shape (Y, 2) with lower and upper bounds of the CI, for
            a metric returning shape (Y,).
        """
        samples = self.bootstrap_metric(
            metric, nb_samples=nb_samples, method=method, ratio=ratio
        )  # (N, Y)
        ci = np.quantile(samples, q=[alpha / 2.0, 1 - alpha / 2.0], axis=0)  # (2, Y)
        ci = np.moveaxis(ci, source=0, destination=-1)  # (Y, 2)
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
