"""
Class to compute metrics for one-to-N matching (identification), such as FPIR, FNIR,
DIR, etc. The class holds scores in a format that allows to compute all metrics
efficiently.
"""

from __future__ import annotations

import operator

import numpy as np

from .scores import BinaryLabel, Scores

OPERATOR_MAP = {
    (BinaryLabel.pos, BinaryLabel.pos): operator.gt,
    (BinaryLabel.pos, BinaryLabel.neg): operator.ge,
    (BinaryLabel.neg, BinaryLabel.pos): operator.lt,
    (BinaryLabel.neg, BinaryLabel.neg): operator.le,
}


class OneToNScores:
    def __init__(
        self,
        neg_rank1,
        pos_rank1,
        pos_mate,
        pos_mate_rank,
        pos_label_rank: np.ndarray | None = None,
        neg_labels: np.ndarray | None = None,
        pos_labels: np.ndarray | None = None,
        score_class: BinaryLabel | str = "neg",
        equal_class: BinaryLabel | str = "pos",
        is_sorted: bool = False,
    ):
        """Scores for one-to-N matching (identification).

        Args:
            neg_rank1: Rank-1 scores or distances for non-mated probes.
            pos_rank1: Rank-1 scores or distances for mated probes.
            pos_mate: For mated probes, score or distance to closest mated element in
                the gallery.
            pos_mate_rank: For mated probes, rank of closest mated element in gallery.
            pos_label_rank: For mated probes, the label-aggregated rank of closest mated
                element in the gallery, i.e., number of distinct non-matching labels in
                lower ranks.
            neg_labels: Labels of non-mated probes. Defaults to assigning each image
                a distinct label. Used for gallery/probe consolidation by label.
            pos_labels: Labels of mated probes. Defaults to assigning each image a
                distinct label. Used for gallery/probe consolidation by label.
            score_class: Do higher scores indicate higher likelyhood of matching
                (``"pos""``, use, e.g., for cosine similarity) or do lower scores
                indicate higher likelyhood of matching (``"neg""``, use, e.g., for
                distances)? Use ``"pos"`` for similarity scores, ``"neg"`` for
                distances.
            equal_class: Do samples with score equal to the threshold get counted as
                a match (``"pos"``) or a non-match (``"neg"``)?
            is_sorted: If True, assume non-mated scores are sorted by ``neg_rank1`` and
                mated scores are sorted by ``pos_mate``.
        """
        self.neg_rank1 = np.asarray(neg_rank1)
        self.pos_rank1 = np.asarray(pos_rank1)
        self.pos_mate = np.asarray(pos_mate)
        self.pos_mate_rank = np.asarray(pos_mate_rank)

        self.pos_label_rank = (
            np.asarray(pos_label_rank) if pos_label_rank is not None else None
        )
        self.neg_labels = np.asarray(neg_labels) if neg_labels is not None else None
        self.pos_labels = np.asarray(pos_labels) if pos_labels is not None else None
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        if not is_sorted:
            sorted_idx = np.argsort(self.neg_rank1)
            self.neg_rank1 = self.neg_rank1[sorted_idx]
            if self.neg_labels is not None:
                self.neg_labels = self.neg_labels[sorted_idx]

            sorted_idx = np.argsort(self.pos_mate)
            self.pos_rank1 = self.pos_rank1[sorted_idx]
            self.pos_mate = self.pos_mate[sorted_idx]
            self.pos_mate_rank = self.pos_mate_rank[sorted_idx]
            if self.pos_label_rank is not None:
                self.pos_label_rank = self.pos_label_rank[sorted_idx]
            if self.pos_labels is not None:
                self.pos_labels = self.pos_labels[sorted_idx]

        # Use threshold function to find matches via
        #    threshold_fn(score, threshold) == True
        self.threshold_fn = OPERATOR_MAP[self.score_class, self.equal_class]

        self.inf_score = -np.inf if self.score_class == BinaryLabel.pos else np.inf

    def __eq__(self, other: OneToNScores) -> bool:
        """Tests if two OneToNScores objects are equal.

        Equality is tested exactly, rounding errors can lead to objects being not equal.
        """

        if not isinstance(other, OneToNScores):
            return NotImplemented

        def _opt_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.array_equal(a, b)

        return (
            np.array_equal(self.neg_rank1, other.neg_rank1)
            and np.array_equal(self.pos_rank1, other.pos_rank1)
            and np.array_equal(self.pos_mate, other.pos_mate)
            and np.array_equal(self.pos_mate_rank, other.pos_mate_rank)
            and _opt_equal(self.pos_label_rank, other.pos_label_rank)
            and _opt_equal(self.neg_labels, other.neg_labels)
            and _opt_equal(self.pos_labels, other.pos_labels)
            and self.score_class == other.score_class
            and self.equal_class == other.equal_class
        )

    def fpir(self, threshold: np.ndarray) -> np.ndarray:
        """False Positive Identification Rate.

        Fraction of non-mated probes for which we falsely identify a match in the
        gallery at the given threshold.

        Args:
            threshold: Threshold or array of thresholds at which to compute FPIR.

        Returns:
            FPIR value(s). Scalar if threshold is scalar, array otherwise.
        """
        return self.to_binary_scores().fpr(threshold)

    def tnir(self, threshold: np.ndarray) -> np.ndarray:
        """True Negative Identification Rate.

        Fraction of non-mated probes for which we correctly identify no matches in the
        gallery at the given threshold. Note that
            TNIR(T) = 1 - FPIR(T)

        Args:
            threshold: Threshold or array of thresholds at which to compute TNIR.

        Returns:
            TNIR value(s). Scalar if threshold is scalar, array otherwise.
        """
        return self.to_binary_scores().tnr(threshold)

    def fnir(
        self, threshold: np.ndarray | None = None, rank: np.ndarray | None = None
    ) -> np.ndarray:
        """False Negative Identification Rate.

        Fraction of mated probes, for which we fail to identify their mate in the
        gallery at the given threshold and within the given rank.

        Args:
            threshold: Threshold or array of thresholds. Can be None if rank is given.
            rank: Maximum rank or array of ranks. Can be None if threshold is given.

        Returns:
            FNIR value(s). Scalar if both threshold and rank are scalar or None, array
            otherwise.
        """
        return 1 - self.tpir(threshold, rank)

    def tpir(
        self, threshold: np.ndarray | None = None, rank: np.ndarray | None = None
    ) -> np.ndarray:
        """True Positive Identification Rate.

        Fraction of mated probes for which we correctly identify their mate in the
        gallery at the given threshold and within the given rank. Note that
            TPIR(T, r) = 1 - FNIR(T, r)

        Args:
            threshold: Threshold or array of thresholds. Can be None if rank is given.
            rank: Maximum rank or array of ranks. Can be None if threshold is given.

        Returns:
            TPIR value(s). Scalar if both threshold and rank are scalar or None, array
            otherwise.
        """
        if threshold is None and rank is None:
            raise ValueError("threshold and rank cannot both be None.")
        if threshold is None:
            return self.dir(rank=rank)
        if rank is None or np.isscalar(rank):
            return self.to_binary_scores(rank).tpr(threshold)

        res = [self.to_binary_scores(rank=r).tpr(threshold) for r in rank]
        res = np.stack(res, axis=-1)
        return res

    def threshold_at_fpir(self, fpir: np.ndarray, *, method="linear") -> np.ndarray:
        """Set threshold at FPIR.

        Args:
            fpir: FPIR values at which to set threshold.
            method: Possible values are ``"linear"``, ``"lower"``, ``"higher"``. If
                ``"lower"`` or ``"higher"``, we return the closest score at which the
                metric is lower or higher than the target. If ``"linear"``, we apply
                linear interpolation between the lower and higher values.

        Returns:
            Threshold value(s) at the given FPIR.
        """
        return self.to_binary_scores().threshold_at_fpr(fpr=fpir, method=method)

    def threshold_at_tnir(self, tnir: np.ndarray, *, method="linear") -> np.ndarray:
        """Set threshold at TNIR.

        Args:
            tnir: TNIR values at which to set threshold.
            method: Possible values are ``"linear"``, ``"lower"``, ``"higher"``. If
                ``"lower"`` or ``"higher"``, we return the closest score at which the
                metric is lower or higher than the target. If ``"linear"``, we apply
                linear interpolation between the lower and higher values.

        Returns:
            Threshold value(s) at the given TNIR.
        """
        return self.to_binary_scores().threshold_at_tnr(tnr=tnir, method=method)

    def threshold_at_fnir(
        self, fnir: np.ndarray, rank: np.ndarray | None = None, *, method="linear"
    ) -> np.ndarray:
        """Set threshold at FNIR within a given rank.

        Args:
            fnir: FNIR values at which to set threshold.
            rank: Maximum rank or array of ranks. If None, no rank constraint is
                applied.
            method: Possible values are ``"linear"``, ``"lower"``, ``"higher"``. If
                ``"lower"`` or ``"higher"``, we return the closest score at which the
                metric is lower or higher than the target. If ``"linear"``, we apply
                linear interpolation between the lower and higher values.

        Returns:
            Threshold value(s) at the given FNIR and rank.
        """

        if rank is None or np.isscalar(rank):
            return self.to_binary_scores(rank).threshold_at_fnr(fnr=fnir, method=method)

        res = [
            self.to_binary_scores(rank=r).threshold_at_fnr(fnr=fnir, method=method)
            for r in rank
        ]
        res = np.stack(res, axis=-1)
        return res

    def threshold_at_tpir(
        self, tpir: np.ndarray, rank: np.ndarray | None = None, *, method="linear"
    ) -> np.ndarray:
        """Set threshold at TPIR within a given rank.

        Args:
            tpir: TPIR values at which to set threshold.
            rank: Maximum rank or array of ranks. If None, no rank constraint is
                applied.
            method: Possible values are ``"linear"``, ``"lower"``, ``"higher"``. If
                ``"lower"`` or ``"higher"``, we return the closest score at which the
                metric is lower or higher than the target. If ``"linear"``, we apply
                linear interpolation between the lower and higher values.

        Returns:
            Threshold value(s) at the given TPIR and rank.
        """

        if rank is None or np.isscalar(rank):
            return self.to_binary_scores(rank).threshold_at_tpr(tpr=tpir, method=method)

        res = [
            self.to_binary_scores(rank=r).threshold_at_tpr(tpr=tpir, method=method)
            for r in rank
        ]
        res = np.stack(res, axis=-1)
        return res

    def mean_rank(
        self, threshold: np.ndarray | None = None, rank: np.ndarray | None = None
    ) -> np.ndarray:
        """Mean Rank for Mated Probes.

        Mean rank of the best correct match in the gallery for mated probes, considering
        only matches below the given threshold and within the given rank. Either or
        both can be None.

        Args:
            threshold: Threshold or array of thresholds. If None, no threshold
                constraint is applied.
            rank: Maximum rank or array of ranks. If None, no rank constraint is
                applied.

        Returns:
            Mean rank value(s). Scalar if both threshold and rank are scalar or None,
            array otherwise.
        """
        scalar_result = (threshold is None or np.isscalar(threshold)) and (
            rank is None or np.isscalar(rank)
        )

        pos_mate_rank = self.pos_mate_rank
        mask = np.full_like(pos_mate_rank, fill_value=True, dtype=np.bool)
        if threshold is not None:
            if np.isscalar(threshold):
                mask &= self.threshold_fn(self.pos_mate, threshold)
            else:
                pos_mate = self.pos_mate[..., None]
                pos_mate_rank = pos_mate_rank[..., None]
                mask = mask[..., None] & self.threshold_fn(pos_mate, threshold)
        pos_mate_rank = np.broadcast_to(pos_mate_rank, mask.shape).copy()

        if rank is not None:
            if np.isscalar(rank):
                mask &= pos_mate_rank <= rank
            else:
                pos_mate_rank = pos_mate_rank[..., None]
                mask = mask[..., None] & (pos_mate_rank <= rank)
        pos_mate_rank = np.broadcast_to(pos_mate_rank, mask.shape).copy()

        pos_mate_rank[~mask] = 0

        s = pos_mate_rank.sum(axis=0)
        n = mask.sum(axis=0)
        mean_rank = np.divide(s, n, out=np.zeros_like(s, dtype=float), where=n != 0)

        if scalar_result:
            mean_rank = mean_rank.item()
        return mean_rank

    def non_match_rate(self, threshold: np.ndarray | None) -> np.ndarray:
        """Non-Match Rate for Mated Probes.

        Proportion of mated probes that have no match (correct or not) in the gallery
        at the given threshold.

        Args:
            threshold: Threshold or array of thresholds at which to compute the
                non-match rate.

        Returns:
            Non-match rate value(s). Scalar if threshold is scalar, array otherwise.
        """
        if threshold is None:
            return 0.0  # No threshold, so all probes have a match

        scores = Scores(
            pos=self.pos_rank1,
            neg=[],
            score_class=self.score_class,
            equal_class=self.equal_class,
            is_sorted=False,
        )
        return scores.fnr(threshold)

    def false_match_rate(self, threshold: np.ndarray, rank: np.ndarray | None = None):
        """False Match Rate for Mated Probes.

        Proportion of mated probes that have a false match in the gallery at the given
        threshold and within the given rank, but no true match. Note the relationship
            FalseMatchRate(T, r) + NonMatchRate(T) = FNIR(T, r)

        Args:
            threshold: Threshold or array of thresholds.
            rank: Maximum rank or array of ranks.

        Returns:
            False match rate value(s). Scalar if both threshold and rank are scalar,
            array otherwise.
        """
        fnir = self.fnir(threshold, rank)
        nmr = self.non_match_rate(threshold)

        if rank is not None and not np.isscalar(rank):
            nmr = np.asarray(nmr)[..., None]

        return fnir - nmr

    def dir(self, rank: np.ndarray | None) -> np.ndarray:
        """Detection and Identification Rate.

        Proportion of mated probes for which a mate is identified in the gallery within
        rank r.

        Args:
            rank: Maximum rank or array of ranks.

        Returns:
            DIR value(s). Scalar if rank is scalar, array otherwise.
        """
        if rank is None:
            return 1.0  # No rank constraint, so all probes are identified

        isscalar = np.isscalar(rank)
        rank = np.asarray([rank] if isscalar else rank)
        dir = (self.pos_mate_rank[:, None] <= rank[None, :]).mean(axis=0)
        if isscalar:
            dir = dir.item()
        return dir

    def consolidate(self, kind: str) -> OneToNScores:
        """Consolidate gallery or probe set or both by label.

        Merge all scores belonging to one label into one, keeping the minimum. This
        corresponds to defining a distance between identities as the smallest distance
        between two images belong to the identities.

        We can consolidate the gallery, the probe set or both.

        Args:
            kind: One of ``"gallery"``, ``"probe"`` or ``"both"``.

        Returns:
            A new OneToNScores object with which we can compute the full set of metrics,
            now at identity level.
        """
        if kind == "gallery" or kind == "both":
            if self.pos_label_rank is None:
                raise ValueError(
                    "Cannot consolidate gallery, if pos_label_rank is None."
                )

            scores = OneToNScores(
                neg_rank1=self.neg_rank1,
                pos_rank1=self.pos_rank1,
                pos_mate=self.pos_mate,
                pos_mate_rank=self.pos_label_rank,
                pos_label_rank=self.pos_label_rank,
                neg_labels=self.neg_labels,
                pos_labels=self.pos_labels,
                score_class=self.score_class,
                equal_class=self.equal_class,
                is_sorted=True,
            )
            if kind == "both":
                scores = scores.consolidate(kind="probe")
            return scores

        if kind != "probe":
            raise ValueError(f"Uknown value for {kind=}.")

        if self.neg_labels is None:
            raise ValueError("Cannot consolidate probes if neg_labels is None.")
        if self.pos_labels is None:
            raise ValueError("Cannot consolidate probes is pos_labels is None.")

        if self.score_class == BinaryLabel.neg:
            _, neg_idx = np.unique(self.neg_labels, return_index=True)
            _, pos_idx = np.unique(self.pos_labels, return_index=True)
        else:
            _, neg_idx = np.unique(self.neg_labels[::-1], return_index=True)
            _, pos_idx = np.unique(self.pos_labels[::-1], return_index=True)
            neg_idx = len(self.neg_labels) - 1 - neg_idx
            pos_idx = len(self.pos_labels) - 1 - pos_idx

        neg_idx.sort()
        pos_idx.sort()

        scores = OneToNScores(
            neg_rank1=self.neg_rank1[neg_idx],
            pos_rank1=self.pos_rank1[pos_idx],
            pos_mate=self.pos_mate[pos_idx],
            pos_mate_rank=self.pos_mate_rank[pos_idx],
            pos_label_rank=(
                self.pos_label_rank[pos_idx]
                if self.pos_label_rank is not None
                else None
            ),
            neg_labels=self.neg_labels[neg_idx],
            pos_labels=self.pos_labels[pos_idx],
            score_class=self.score_class,
            equal_class=self.equal_class,
            is_sorted=True,
        )
        return scores

    def rank_one_scores(self, subset: str = "all") -> np.ndarray:
        """Returns all rank one scores (mated, non-mated or both).

        Can be used to analyse score distributions.

        Args:
            subset: One of ``"mated"``, ``"non_mated"`` or ``"all"``.

        Returns:
            Rank one scores for the given subset. Scores are not guaranteed to be
            sorted, in particular when ``subset="all"``.
        """
        if subset == "mated":
            return self.pos_rank1
        elif subset == "non_mated":
            return self.neg_rank1
        elif subset == "all":
            return np.concat([self.pos_rank1, self.neg_rank1])
        else:
            raise ValueError(f"Unknown value for {subset=}.")

    def to_binary_scores(self, rank: int | None = None) -> Scores:
        """Converts 1:N matching into a binary classification problem.

        Takes the best mated score for mated probes as positive class scores and the
        best score for non-mated probes as negative class scores. This measures the
        ability to identify mated matches against the cost of non-mated matches, while
        ignoring the cost of missed or false matches for mated probes.

        Mated probes whose best mate is not within the given rank are counted as false
        negatives (distance is set to Inf).

        Args:
            rank: Maximum rank. Mated probes whose best mate has a rank above this
                value are treated as unmatched. If None, no rank constraint is applied.

        Returns:
            A ``Scores`` object for binary classification.
        """
        if rank is not None:
            # Filter by rank
            mask = self.pos_mate_rank <= rank
            finite = self.pos_mate[mask]
            nb_inf = len(self.pos_mate) - len(finite)

            # Add infinite scores at beginning or end to maintain sort order
            if self.score_class == BinaryLabel.pos:
                pos = np.concat([np.full(nb_inf, fill_value=-np.inf), finite])
            else:
                pos = np.concat([finite, np.full(nb_inf, fill_value=np.inf)])
        else:
            pos = self.pos_mate

        return Scores(
            pos=pos,
            neg=self.neg_rank1,
            score_class=self.score_class,
            equal_class=self.equal_class,
            is_sorted=True,
        )
