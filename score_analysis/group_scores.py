"""
The :class:`GroupScores` class is an extension of the :class:`.Scores` class to also
keep track of group membership.

 * We only support one group. To perform analysis with respect to multiple groups
   simultaneously, one can create a new group variable that combines all original
   groups.
 * All per-group metric functions will return arrays with :math:`G` as the first
   dimension, where :math:`G` is the number of different groups. We can use
   ``GroupScores.groups`` to correlate the array indices with group names. Arrays will
   play nicely with the existing vectorisation features in the library.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np

from .cm import ConfusionMatrix
from .scores import (
    DEFAULT_BOOTSTRAP_CONFIG,
    SAMPLING_METHOD_DYNAMIC,
    SAMPLING_METHOD_PROPORTION,
    SAMPLING_METHOD_REPLACEMENT,
    SAMPLING_METHOD_SINGLE_PASS,
    SINGLE_PASS_SAMPLE_THRESHOLD,
    BinaryLabel,
    BootstrapConfig,
    SamplingMethod,
    Scores,
)


def groupwise(metric) -> Callable[..., np.ndarray]:
    """
    Converts a metric to a group-wise metric. A metric is can be a string indicating a
    member function of the :class:`.Scores` class or a callable with signature::

        metric(sample: Scores, **kwargs) -> np.ndarray

    The group-wise metric is a function::

        groupwise_metric(sample: GroupScores, **kwargs) -> np.ndarray

    returning an array of shape :math:`(G, Y)`, where :math:`G` is the number of groups
    and metric returns an array of shape :math:`Y`.
    """
    if isinstance(metric, str):
        metric = getattr(Scores, metric)

    def groupwise_metric(scores, **kwargs):
        res = [metric(scores[group], **kwargs) for group in scores.groups]
        res = np.stack(res, axis=0)
        return res

    return groupwise_metric


class GroupScores(Scores):
    def __init__(
        self,
        pos,
        neg,
        *,
        pos_groups,
        neg_groups,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
        group_names: Optional[Any] = None,
        is_sorted: bool = False,
    ):
        """
        Args:
            pos: Scores for positive samples.
            neg: Scores for negative samples.
            pos_groups: Group labels for positive samples.
            neg_groups: Group labels for negative samples.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?
            group_names: Explicitly provided group names. If provided, this list will
                be used as is and not sorted.
            is_sorted: If True, we assume the scores are already sorted. Can be used to
                speed up initialisation.
        """
        super().__init__(
            pos=pos,
            neg=neg,
            nb_easy_pos=0,  # We don't support easy samples.
            nb_easy_neg=0,
            score_class=score_class,
            equal_class=equal_class,
            is_sorted=True,  # We have to sort ourselves to keep group labels in sync.
        )

        self.pos_groups = np.asarray(pos_groups)
        self.neg_groups = np.asarray(neg_groups)

        if not is_sorted:
            pos_idx = np.argsort(self.pos)
            neg_idx = np.argsort(self.neg)

            self.pos = self.pos[pos_idx]
            self.neg = self.neg[neg_idx]
            self.pos_groups = self.pos_groups[pos_idx]
            self.neg_groups = self.neg_groups[neg_idx]

        # This is a (sorted) array of all group names. Having this allows us to return
        # arrays in functions such as group_far() and correlate the array entries with
        # group names.
        if group_names is None:
            self.groups = np.asarray(sorted(set(pos_groups) | set(neg_groups)))
        else:
            self.groups = np.asarray(group_names)

        # This dictionary contains Scores objects for each group individually. They are
        # only computed, when required. Use __getitem__ to access them.
        self._grouped_scores = {}

    @staticmethod
    def from_labels(
        labels,
        scores,
        groups,
        *,
        pos_label: Any = 1,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
        is_sorted: bool = False,
    ) -> GroupScores:
        """
        Args:
            labels: Array with sample labels.
            scores: Array with sample scores.
            groups: Array with group labels.
            pos_label: The label of the positive class. All other labels are treated as
                negative labels.
            score_class: Do scores indicate membership of the positive or the negative
                class?
            equal_class: Do samples with score equal to the threshold get assigned to
                the positive or negative class?
            is_sorted: If True, we assume the scores are already sorted. Can be used to
                speed up initialisation.

        Returns:
            A :class:`GroupScores` instance.
        """
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        groups = np.asarray(groups)
        return GroupScores(
            pos=scores[labels == pos_label],
            neg=scores[labels != pos_label],
            pos_groups=groups[labels == pos_label],
            neg_groups=groups[labels != pos_label],
            score_class=score_class,
            equal_class=equal_class,
            is_sorted=is_sorted,
        )

    def __eq__(self, other: GroupScores) -> bool:
        """
        Tests if two :class:`GroupScores` objects are equal. Equality is tested
        exactly, rounding errors can lead to objects being not equal.

        Args:
            other: Object to test against.

        Returns:
            True, if objects are equal, false otherwise.
        """
        equal = (
            np.array_equal(self.pos, other.pos)
            and np.array_equal(self.neg, other.neg)
            and np.array_equal(self.pos_groups, other.pos_groups)
            and np.array_equal(self.neg_groups, other.neg_groups)
            and self.nb_easy_pos == other.nb_easy_pos  # Although not used, we keep it.
            and self.nb_easy_neg == other.nb_easy_neg
            and self.score_class == other.score_class
            and self.equal_class == other.equal_class
        )
        return equal

    def swap(self) -> GroupScores:
        """
        Swaps positive and negative scores. Also reverses the decision logic, so that
        fpr of original scores equals fnr of reversed scores.

        Returns:
            :class:`GroupScores` object with positive and negative scores reversed.
        """
        return GroupScores(
            pos=self.neg,
            neg=self.pos,
            pos_groups=self.neg_groups,
            neg_groups=self.pos_groups,
            score_class="neg" if self.score_class == BinaryLabel.pos else "pos",
            equal_class="neg" if self.equal_class == BinaryLabel.pos else "pos",
            is_sorted=True,
        )

    def __getitem__(self, group) -> Scores:
        """
        Access the :class:`.Scores` for a particular subgroup. We cache the results for
        fast repeat access, e.g., when computing multiple group-wise metrics.

        Args:
            group: Group, whose scores to access.

        Returns:
            :class:`.Scores` object for the group.
        """
        if group not in self.groups:
            raise ValueError(f"Group {group} does not exist in the scores object.")
        if group not in self._grouped_scores:
            self._grouped_scores[group] = Scores(
                pos=self.pos[self.pos_groups == group],
                neg=self.neg[self.neg_groups == group],
                nb_easy_neg=0,
                nb_easy_pos=0,
                score_class=self.score_class,
                equal_class=self.equal_class,
                is_sorted=True,
            )
        return self._grouped_scores[group]

    def group_cm(self, threshold) -> ConfusionMatrix:
        """
        Computes per-group confusion matrices at the given threshold(s).

        Args:
            threshold: Can be a scalar or array-like.

        Returns:
            A binary :class:`.ConfusionMatrix` of shape (G, T, 2, 2), where G is number
            of groups and T is shape of threshold array.
        """
        cm = [np.asarray(self[group].cm(threshold)) for group in self.groups]
        cm = np.stack(cm, axis=0)
        return ConfusionMatrix(matrix=cm, binary=True)

    def group_tpr(self, threshold) -> np.ndarray:
        """Per-group True Positive Rate at threshold(s)."""
        return self.group_cm(threshold).tpr()

    def group_fnr(self, threshold):
        """Per-group False Negative Rate at threshold(s)."""
        return self.group_cm(threshold).fnr()

    def group_tnr(self, threshold):
        """Per-group True Negative Rate at threshold(s)."""
        return self.group_cm(threshold).tnr()

    def group_fpr(self, threshold):
        """Per-group False Positive Rate at threshold(s)."""
        return self.group_cm(threshold).fpr()

    def group_topr(self, threshold):
        """Per-group Test Outcome Positive Rate at threshold(s)."""
        return self.group_cm(threshold).topr()

    def group_tonr(self, threshold):
        """Per-group Test Outcome Negative Rate at threshold(s)."""
        return self.group_cm(threshold).tonr()

    # Aliases.
    def group_tar(self, threshold):
        """
        Per-group True Acceptance Rate at threshold(s).

        Alias for :func:`~GroupScores.group_tpr`.
        """
        return self.group_tpr(threshold)

    def group_frr(self, threshold):
        """
        Per-group False Rejection Rate at threshold(s).

        Alias for :func:`~GroupScores.group_fnr`.
        """
        return self.group_fnr(threshold)

    def group_trr(self, threshold):
        """
        Per-group True Rejection Rate at threshold(s).

        Alias for :func:`~GroupScores.group_tnr`.
        """
        return self.group_tnr(threshold)

    def group_far(self, threshold):
        """
        Per-group False Acceptance Rate at threshold(s).

        Alias for :func:`~GroupScores.group_fpr`.
        """
        return self.group_fpr(threshold)

    def group_acceptance_rate(self, threshold):
        """
        Per-group Acceptance Rate at threshold(s).

        Alias for :func:`~GroupScores.group_topr`.
        """
        return self.group_topr(threshold)

    def group_rejection_rate(self, threshold):
        """
        Per-group Rejection Rate at threshold(s).

        Alias for :func:`~GroupScores.group_tonr`.
        """
        return self.group_tonr(threshold)

    def _sampling_method(self, config: BootstrapConfig) -> SamplingMethod:
        """
        If sampling method is "dynamic", we select the appropriate method between
        replacement and single pass, depending on the other bootstrap parameters.
        """
        if config.sampling_method != SAMPLING_METHOD_DYNAMIC:
            return config.sampling_method  # Nothing to choose here.

        if config.stratified_sampling == "by_group":
            # With group-wise stratification single pass sampling is not faster.
            return SAMPLING_METHOD_REPLACEMENT
        elif (
            self.nb_hard_pos < SINGLE_PASS_SAMPLE_THRESHOLD
            or self.nb_hard_neg < SINGLE_PASS_SAMPLE_THRESHOLD
        ):
            return SAMPLING_METHOD_REPLACEMENT
        else:
            return SAMPLING_METHOD_SINGLE_PASS

    def bootstrap_sample(
        self,
        config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
    ) -> GroupScores:
        """
        Creates one bootstrap sample by sampling with the specified configuration.

        Args:
            config: Bootstrap configuration.

        Returns:
            :class:`GroupScores` object with the sampled scores.
        """
        if config.smoothing:
            raise ValueError("Bootstrap smoothing is not implemented for GroupScores.")

        sampling_method = self._sampling_method(config)

        replacement_sampling = sampling_method == SAMPLING_METHOD_REPLACEMENT
        single_pass_sampling = sampling_method == SAMPLING_METHOD_SINGLE_PASS

        if replacement_sampling or single_pass_sampling:
            stratified_sampling = config.stratified_sampling
            if stratified_sampling == "by_label" or not stratified_sampling:
                pos_idx, neg_idx, _, _ = self._sample_indices(
                    by_label=stratified_sampling == "by_label",
                    single_pass=single_pass_sampling,
                )
                scores = GroupScores(
                    pos=self.pos[pos_idx],
                    neg=self.neg[neg_idx],
                    pos_groups=self.pos_groups[pos_idx],
                    neg_groups=self.neg_groups[neg_idx],
                    score_class=self.score_class,
                    equal_class=self.equal_class,
                    group_names=self.groups,
                    # Single-pass sampling returns already sorted scores
                    is_sorted=single_pass_sampling,
                )
            elif stratified_sampling == "by_group":
                # Note: Single pass sampling can technically be combined with
                # group-wise stratification, but it doesn't lead to faster sampling,
                # since after the group-wise sampling stage the scores for each group
                # are concatenated and then still need to be sorted.
                pos, neg, pos_groups, neg_groups = [], [], [], []
                for group in self.groups:
                    group_scores = self[group]
                    pos_idx, neg_idx, _, _ = group_scores._sample_indices(
                        by_label=False, single_pass=single_pass_sampling
                    )
                    pos.append(group_scores.pos[pos_idx])
                    neg.append(group_scores.neg[neg_idx])
                    pos_groups.append(np.asarray([group for _ in pos_idx]))
                    neg_groups.append(np.asarray([group for _ in neg_idx]))

                scores = GroupScores(
                    pos=np.concatenate(pos),
                    neg=np.concatenate(neg),
                    pos_groups=np.concatenate(pos_groups),
                    neg_groups=np.concatenate(neg_groups),
                    score_class=self.score_class,
                    equal_class=self.equal_class,
                    group_names=self.groups,
                    is_sorted=False,  # See note above about single pass sampling here.
                )
            else:
                raise ValueError(f"Unsupported value for {stratified_sampling=}.")

            if config.smoothing:
                raise ValueError("Smoothing is not supported for GroupScores.")

        elif sampling_method == SAMPLING_METHOD_PROPORTION:
            raise ValueError("Proportional sampling is not supported for GroupScores.")
        elif isinstance(sampling_method, str):
            raise ValueError(f"Unsupported sampling method {config.sampling_method}.")
        elif callable(sampling_method):
            scores = sampling_method(self)  # Custom sampling method
        else:
            raise ValueError("Sampling method must be a string or a callable.")

        return scores
