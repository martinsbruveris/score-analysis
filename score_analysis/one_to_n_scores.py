"""
This is brainstorming and notes for later...

First, I need to set the threshold. Assuming the threshold is set via FPIR, I need
to know:
  - For each probe x_i, which is the closest element from the gallery y_i.

I want a data structure that allows me to compute FPIR(T), FNIR(T) efficiently for
multiple values of T. Ideally this should be vectorised.

I want both image level values, as well as aggregates by identity. This means I need to
store all matches below a certain threshold.

How do I know the threshold? I can use 1:1 matching to give me an idea, how the
threshold relates to memory required to store all matches below that threshold. E.g.,
with 100k probes and 1m gallery, I have 1e11 comparisons; setting the threshold at
FMR of 1e-4, given 1e7 matches, which requires <100MB space.

So, I have limits
  - Threshold in [0, T_max]
  - FPIR(T) in [0, FPIR(T_max)]
  - FNIR(T) in [FNIR(T_max), 1] (note, decreasing in T)

What data structure do I want?
  - Given T, I want to know for each probe image, how many gallery samples have dist<T.
  - To aggregate at identity level, I will also want the min by label. But this can
    be accomplished by an aggregation step.
  - The two options I have are:
      - Sort by distance and leave unsorted by label

What do I want to be able to measure?
  - FPIR(T). Need: For each non-mated probe, distance to closest element in gallery.
  - FNIR(T, r). Need: For each mated probe, distance to closest mated element in gallery
    if withing rank r, Inf otherwise. Sufficient: For each mated probe, distance to
    closest mated element in gallery and rank of closest mated element in gallery.
  - MeanRank. Need: For each mated probe, rank of closest mated element in gallery.
  - NonMatchRate(T). Proportion of mated probes that have no match (correct or not) at
    threshold T. Need: For each mated probe, all rank 1 distances.
  - FalseMatchRate(T, r). Proportion of mated probes that have a false match at
    threshold T and within rank r. Note that:
        FalseMatchRate(T, r) + NonMatchRate(T) = FNIR(T, r)

This is a 1:1 metric masquerading a 1:N metric.
  - MeanFalseMatches(T). Avegage number of false matches (mated or non-mated) at
    threshold T. Need: For exact results, need all matches up to threshold T.

These are the items I need
  - neg_rank1_dist
  - pos_rank1_dist
  - mate_rank
  - mate_dist
  - id_rank
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

    def fpir(self, threshold):
        """False Positive Identification Rate

        Fraction of non-mated probes for which we falsely identify a match in the
        gallery at the given threshold.
        """
        return self.to_binary_scores().fpr(threshold)

    def tnir(self, threshold):
        """True Negative Identification Rate

        Fraction of non-mated probes for which we correctly identify no matches in the
        gallery at the given threshold. Note that
            TNIR(T) = 1 - FPIR(T)
        """
        return self.to_binary_scores().tnr(threshold)

    def fnir(self, threshold: np.ndarray | None = None, rank: np.ndarray | None = None):
        """False Negative Identification Rate

        Fraction of mated probes, for which we fail to identify their mate in the
        gallery at the given threshold and within the given rank.

        One of threshold or rank can be None (but not both).
        """
        return 1 - self.tpir(threshold, rank)

    def tpir(self, threshold: np.ndarray | None = None, rank: np.ndarray | None = None):
        """True Positive Identification Rate

        Fraction of mated probes for which we correctly identify their mate in the
        gallery at the given threshold and within the given rank. Note that
            TPIR(T, r) = 1 - FNIR(T, r)
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

    def threshold_at_fpir(self, fpir, *, method="linear"):
        """Set threshold at FPIR

        Args:
            fpir: FPIR values at which to set threshold.
            method: Possible values are "linear", "lower", "higher". If "lower"
                or "higher", we return the closest score at which the metric is
                lower or higher that the target. If "linear", we apply linear
                interpolation between the lower and higher values.
        """
        return self.to_binary_scores().threshold_at_fpr(fpr=fpir, method=method)

    def threshold_at_tnir(self, tnir, *, method="linear"):
        """Set threshold at TNIR"""
        return self.to_binary_scores().threshold_at_tnr(tnr=tnir, method=method)

    def threshold_at_fnir(self, fnir, rank=None, *, method="linear"):
        """Set threshold at FNIR within a given rank."""

        if rank is None or np.isscalar(rank):
            return self.to_binary_scores(rank).threshold_at_fnr(fnr=fnir, method=method)

        res = [
            self.to_binary_scores(rank=r).threshold_at_fnr(fnr=fnir, method=method)
            for r in rank
        ]
        res = np.stack(res, axis=-1)
        return res

    def threshold_at_tpir(self, tpir, rank=None, *, method="linear"):
        """Set threshold at TPIR within a given rank."""

        if rank is None or np.isscalar(rank):
            return self.to_binary_scores(rank).threshold_at_tpr(tpr=tpir, method=method)

        res = [
            self.to_binary_scores(rank=r).threshold_at_tpr(tpr=tpir, method=method)
            for r in rank
        ]
        res = np.stack(res, axis=-1)
        return res

    def mean_rank(self, threshold=None, rank=None):
        """Mean Rank for Mated Probes.

        Mean rank of the best correct match in the gallery for mated probes, considering
        only matches below the given threshold and within the given rank. Either or
        both can be None.
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

    def non_match_rate(self, threshold):
        """Non-Match Rate for Mated Probes

        Proportion of mated probes that have no match (correct or not) in the gallery
        at the given threshold.
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

    def false_match_rate(self, threshold, rank):
        """False Match Rate for Mated Probes

        Proportion of mated probes that have a false match in the gallery at the given
        threshold and within the given rank, but no true match. Note the relationship
            FalseMatchRate(T, r) + NonMatchRate(T) = FNIR(T, r)
        """
        fnir = self.fnir(threshold, rank)
        nmr = self.non_match_rate(threshold)

        if rank is not None and not np.isscalar(rank):
            nmr = np.asarray(nmr)[..., None]

        return fnir - nmr

    def dir(self, rank: np.ndarray):
        """Detection and Identification Rate

        Proportion of mated probes for which a mate is identified in the gallery within
        rank r.
        """
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
        """Converts 1:N matching into a binary classification problem, by taking the
        best mated score for mated probes for positive class scores and the best score
        for non-mated probes for negative class scores.

        It measures the ability to identify mated matches against the cost of non-mated
        matches, while ignoring the cost of missed or false matches for mated probes.

        Mated probes whose best mate is not within the given rank are counted as false
        negatives (distance is set to Inf).
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
