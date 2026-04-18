"""
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
"""

from __future__ import annotations

import operator
from typing import Union

import numpy as np

from score_analysis.scores import BinaryLabel

OPERATOR_MAP = {
    ("pos", "pos"): operator.lt,
    ("pos", "neg"): operator.le,
    ("neg", "pos"): operator.le,
    ("neg", "neg"): operator.lt,
}


class OneToNScores:
    def __init__(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
        pos_idx: np.ndarray,
        neg_idx: np.ndarray,
        pos_labels: np.ndarray,
        neg_labels: np.ndarray,
        gallery_labels: np.ndarray,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
        is_sorted: bool = False,
    ):
        self.pos = np.asarray(pos)  # (Kp, r)  mated distances
        self.neg = np.asarray(neg)  # (Kn, r)  non-mated distances
        self.pos_idx = np.asarray(pos_idx)  # (Kp, r) indices
        self.neg_idx = np.asarray(neg_idx)  # (Kn, r) indices
        self.pos_labels = np.asarray(pos_labels)
        self.neg_labels = np.asarray(neg_labels)
        self.gallery_labels = np.asarray(gallery_labels)
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        self.threshold_fn = OPERATOR_MAP[
            (self.score_class.value, self.equal_class.value)
        ]

        if not is_sorted:
            # Sort each row of pos and neg independently.
            # If score_class is "pos", sort in descending order, otherwise ascending.
            self.pos, self.pos_idx = self._sort_rows(self.pos, self.pos_idx)
            self.neg, self.neg_idx = self._sort_rows(self.neg, self.neg_idx)

    def _sort_rows(self, scores, idx):
        if scores.ndim < 2 or scores.shape[-1] == 0:
            return scores, idx
        order = np.argsort(scores, axis=-1)
        if self.score_class == BinaryLabel.pos:
            order = order[:, ::-1]
        return (
            np.take_along_axis(scores, order, axis=-1),
            np.take_along_axis(idx, order, axis=-1),
        )

    def __eq__(self, other: OneToNScores) -> bool:
        """
        Tests if two OneToNScores objects are equal. Equality is tested exactly,
        rounding errors can lead to objects being not equal.

        Args:
            other: Object to test against.

        Returns:
            True, if objects are equal, false otherwise.
        """
        if not isinstance(other, OneToNScores):
            return NotImplemented
        equal = (
            np.array_equal(self.pos, other.pos)
            and np.array_equal(self.neg, other.neg)
            and np.array_equal(self.pos_idx, other.pos_idx)
            and np.array_equal(self.neg_idx, other.neg_idx)
            and np.array_equal(self.pos_labels, other.pos_labels)
            and np.array_equal(self.neg_labels, other.neg_labels)
            and np.array_equal(self.gallery_labels, other.gallery_labels)
            and self.score_class == other.score_class
            and self.equal_class == other.equal_class
        )
        return equal

    @staticmethod
    def from_matrix(
        matrix: np.ndarray,
        probes: np.ndarray,
        gallery: np.ndarray,
        rank: int = 1,
        score_class: Union[BinaryLabel, str] = "pos",
        equal_class: Union[BinaryLabel, str] = "pos",
    ):
        matrix = np.asarray(matrix, dtype=float)  # (P, G)
        probes = np.asarray(probes)  # (P,)
        gallery = np.asarray(gallery)  # (G,)

        # Sort each probe's scores across gallery, best matches first.
        # For score_class="pos": descending (higher is better).
        # For score_class="neg": ascending (lower is better).
        sort_idx = np.argsort(matrix, axis=1)
        if BinaryLabel(score_class) == BinaryLabel.pos:
            sort_idx = sort_idx[:, ::-1]

        # Keep only top-r matches
        sort_idx = sort_idx[:, :rank]
        sorted_scores = np.take_along_axis(matrix, sort_idx, axis=1)
        sorted_gallery = gallery[sort_idx]

        # Split into mated (pos) and non-mated (neg) probes
        is_mated = np.isin(probes, gallery)

        return OneToNScores(
            pos=sorted_scores[is_mated],
            neg=sorted_scores[~is_mated],
            pos_idx=sorted_gallery[is_mated],
            neg_idx=sorted_gallery[~is_mated],
            pos_labels=probes[is_mated],
            neg_labels=probes[~is_mated],
            gallery_labels=gallery,
            score_class=score_class,
            equal_class=equal_class,
            is_sorted=True,
        )

    @staticmethod
    def from_embeddings(
        probe_emb: np.ndarray,
        gallery_emb: np.ndarray,
        probe_labels: np.ndarray,
        gallery_label: np.ndarray,
        dist: str = "l2_squared",
        rank: int = 1,
        equal_class: Union[BinaryLabel, str] = "neg",
    ): ...

    def fpir(self, threshold: np.ndarray) -> np.ndarray: ...

    def fnir(self, threshold: np.ndarray) -> np.ndarray: ...
