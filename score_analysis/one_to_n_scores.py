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
from typing import TYPE_CHECKING

import numpy as np

from .embeddings import probe_gallery_distances
from .scores import BinaryLabel, Scores

if TYPE_CHECKING:  # pragma: no cover
    import torch


OPERATOR_MAP = {
    (BinaryLabel.pos, BinaryLabel.pos): operator.le,
    (BinaryLabel.pos, BinaryLabel.neg): operator.lt,
    (BinaryLabel.neg, BinaryLabel.pos): operator.lt,
    (BinaryLabel.neg, BinaryLabel.neg): operator.le,
}


class OneToNScores:
    def __init__(
        self,
        neg_rank1_dist,
        pos_rank1_dist,
        pos_mate_dist,
        pos_mate_rank,
        pos_id_rank: np.ndarray | None = None,
        pos_labels: np.ndarray | None = None,
        score_class: BinaryLabel | str = "neg",
        equal_class: BinaryLabel | str = "neg",
    ):
        """Scores for one-to-N matching (identification).

        Args:
            neg_rank1_dist: Rank-1 distances for non-mated probes.
            pos_rank1_dist: Rank-1 distances for mated probes.
            pos_mate_dist: For mated probes, distance to closest mated element in
                gallery.
            pos_mate_rank: For mated probes, distance to clostes 
            score_class: Do higher scores indicate higher likelyhood of matching
                (``"pos""``, use, e.g., for cosine similarity) or do lower scores
                indicate higher likelyhood of matching (``"neg""``, use, e.g., for
                distances)? Use ``"pos"`` for similarity scores, ``"neg"`` for
                distances.
            equal_class: Do samples with score equal to the threshold get counted as
                a match (``"pos"``) or a non-match (``"neg"``)?
            is_sorted: If True, assume each row is already sorted best-first.
        """
        self.neg_rank1_dist = np.asarray(neg_rank1_dist)
        self.pos_rank1_dist = np.asarray(pos_rank1_dist)
        self.pos_mate_dist = np.asarray(pos_mate_dist)
        self.pos_mate_rank = np.asarray(pos_mate_rank)
        self.pos_id_rank = pos_id_rank or pos_mate_rank.copy()
        self.pos_labels = pos_labels or np.arange(len(pos_rank1_dist))
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        # Use threshold function to find matches via
        #    threshold_fn(score, threshold) == True
        self.threshold_fn = OPERATOR_MAP[self.score_class, self.equal_class]

    def fpir(self, threshold):
        """False Positive Identification Rate

        Fraction of non-mated probes for which we falsely identify a match in the
        gallery at the given threshold.
        """

    def tnir(self, threshold):
        """True Negative Identification Rate

        Fraction of non-mated probes for which we correctly identify no matches in the
        gallery at the given threshold. Note that
            TNIR(T) = 1 - FPIR(T)
        """

    def fnir(self, threshold, rank):
        """False Negative Identification Rate

        Fraction of mated probes, for which we fail to identify their mate in the
        gallery at the given threshold and within the given rank.

        One of threshold or rank can be None (but not both).
        """

    def tpir(self, threshold, rank):
        """True Positive Identification Rate

        Fraction of mated probes for which we correctly identify their mate in the
        gallery at the given threshold and within the given rank. Note that
            TPIR(T, r) = 1 - FNIR(T, r)
        """

    def threshold_at_fpir(self, fpir):
        """Set threshold at FPIR"""

    def threshold_at_tnir(self, tnir):
        """Set threshold at TNIR"""

    def threshold_at_fnir(self, fnir, rank):
        """Set threshold at FNIR within a given rank."""

    def threshold_at_tpir(self, tpir, rank):
        """Set threshold at TPIR within a given rank."""

    def mean_rank(self, threshold=None, rank=None):
        """Mean Rank for Mated Probes

        Mean rank of the best correct match in the gallery for mated probes, considering
        only matches below the given threshold and within the given rank. Either or
        both can be None.
        """

    def non_match_rate(self, threshold):
        """Non-Match Rate for Mated Probes

        Proportion of mated probes that have no match (correct or not) in the gallery
        at the given threshold.
        """

    def false_match_rate(self, threshold, rank):
        """False Match Rate for Mated Probes

        Proportion of mated probes that have a false match in the gallery at the given
        threshold and within the given rank. Note the relationship
            FalseMatchRate(T, r) + NonMatchRate(T) = FNIR(T, r)
        """

    def dir(self, rank):
        """Detection and Identification Rate

        Proportion of mated probes for which a mate is identified in the gallery within
        rank r.
        """
        return self.tpir(threshold=None, rank=rank)

    def consolidate(self, kind: str) -> OneToNScores:
        """Consolidate gallery or probe set or both by identity

        Merge all scores belonging to one identity into one, keeping the minimum. This
        corresponds to defining a distance between identities as the smallest distance
        between two images belong to the identities.

        We can consolidate the gallery, the probe set or both.

        Returns:
            A new OneToNScores object with which we can compute the full set of metrics,
            now at identity level.
        """

    def rank_one_scores(self, subset: str):
        """Returns all rank one scores (mated, non-mated or both).

        Can be used to analyse score distributions.
        """

    def to_binary_scores(self, rank) -> Scores:
        """Converts 1:N matching into a binary classification problem, by taking the
        best mated score for mated probes for positive class scores and the best score
        for non-mated probes for negative class scores.

        It measures the ability to identify mated matches against the cost of non-mated
        matches, while ignoring the cost of missed or false matches for mated probes.

        Mated probes whose best mate is not within the given rank are counted as false
        negatives.
        """
