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
        pos: np.ndarray,
        neg: np.ndarray,
        pos_idx: np.ndarray,
        neg_idx: np.ndarray,
        pos_labels: np.ndarray,
        neg_labels: np.ndarray,
        gallery_labels: np.ndarray,
        score_class: BinaryLabel | str = "pos",
        equal_class: BinaryLabel | str = "pos",
        is_sorted: bool = False,
    ):
        """Scores for one-to-N matching (identification).

        Stores per-probe scores against the top-ranked gallery items, split into
        mated probes (identity exists in gallery) and non-mated probes (identity
        does not exist in gallery). Rows are sorted so that the best match comes
        first (descending for ``score_class="pos"``, ascending for
        ``score_class="neg"``).

        Args:
            pos: Array of shape ``(Kp, rank)`` with scores for mated probes.
            neg: Array of shape ``(Kn, rank)`` with scores for non-mated probes.
            pos_idx: Array of shape ``(Kp, rank)`` with indices of gallery labels
                corresponding to each score in ``pos``.
            neg_idx: Array of shape ``(Kn, rank)`` with indices of gallery labels
                corresponding to each score in ``neg``.
            pos_labels: Array of shape ``(Kp,)`` with probe labels for mated
                probes.
            neg_labels: Array of shape ``(Kn,)`` with probe labels for non-mated
                probes.
            gallery_labels: Array of shape ``(G,)`` with gallery identity labels.
            score_class: Do higher scores indicate higher likelyhood of matching
                (``"pos""``, use, e.g., for cosine similarity) or do lower scores
                indicate higher likelyhood of matching (``"neg""``, use, e.g., for
                distances)? Use ``"pos"`` for similarity scores, ``"neg"`` for
                distances.
            equal_class: Do samples with score equal to the threshold get counted as
                a match (``"pos"``) or a non-match (``"neg"``)?
            is_sorted: If True, assume each row is already sorted best-first.
        """
        self.pos = np.asarray(pos)  # (Kp, r)  mated distances
        self.neg = np.asarray(neg)  # (Kn, r)  non-mated distances
        self.pos_idx = np.asarray(pos_idx)  # (Kp, r) indices
        self.neg_idx = np.asarray(neg_idx)  # (Kn, r) indices
        self.pos_labels = np.asarray(pos_labels)
        self.neg_labels = np.asarray(neg_labels)
        self.gallery_labels = np.asarray(gallery_labels)
        self.score_class = BinaryLabel(score_class)
        self.equal_class = BinaryLabel(equal_class)

        # Use threshold function to find matches via
        #    threshold_fn(score, threshold) == True
        self.threshold_fn = OPERATOR_MAP[self.score_class, self.equal_class]

        if not is_sorted:
            # Sort each row of pos and neg independently.
            # If score_class is "pos", sort in descending order, otherwise ascending.
            self.pos, self.pos_idx = self._sort_rows(self.pos, self.pos_idx)
            self.neg, self.neg_idx = self._sort_rows(self.neg, self.neg_idx)

            self._sort_neg()

    def _sort_rows(self, scores, idx):
        if scores.ndim < 2:
            raise ValueError("Scores must be 2D matrix.")
        if scores.shape[-1] == 0:
            return scores, idx
        order = np.argsort(scores, axis=-1)
        if self.score_class == BinaryLabel.pos:
            order = order[:, ::-1]
        return (
            np.take_along_axis(scores, order, axis=-1),
            np.take_along_axis(idx, order, axis=-1),
        )

    def _sort_neg(self):
        """Sort negative scores by first score. Used for efficient FPIR computation."""
        if self.neg.shape[0] == 0:
            return
        order = np.argsort(self.neg[:, 0])
        self.neg = self.neg[order]
        self.neg_idx = self.neg_idx[order]
        self.neg_labels = self.neg_labels[order]

    def __str__(self) -> str:
        Kp, r = self.pos.shape
        Kn = self.neg.shape[0]
        G = len(self.gallery_labels)
        n = 10  # Max rows to display
        with np.printoptions(threshold=n * r, edgeitems=n // 2):
            lines = [
                f"OneToNScores("
                f"mated={Kp}, non_mated={Kn}, gallery={G}, rank={r}, "
                f"score_class={self.score_class.value}, "
                f"equal_class={self.equal_class.value})",
                f"  pos={self.pos[:n]}",
                f"  pos_idx={self.pos_idx[:n]}",
                f"  pos_labels={self.pos_labels[:n]}",
                f"  neg={self.neg[:n]}",
                f"  neg_idx={self.neg_idx[:n]}",
                f"  neg_labels={self.neg_labels[:n]}",
                f"  gallery_labels={self.gallery_labels[:n]}",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

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
        probe_labels: np.ndarray,
        gallery_labels: np.ndarray,
        rank: int | None = 1,
        score_class: BinaryLabel | str = "pos",
        equal_class: BinaryLabel | str = "pos",
    ):
        """Construct from a precomputed probe-gallery score matrix. We assume that this
        is the full matrix of scores.

        Sorts each probe's scores across the gallery, keeps the top ``rank``
        matches, and splits probes into mated and non-mated.

        Args:
            matrix: Score matrix of shape ``(P, G)`` where entry ``(i, j)`` is
                the score between probe ``i`` and gallery item ``j``.
            probe_labels: Array of shape ``(P,)`` with identity labels for each
                probe.
            gallery_labels: Array of shape ``(G,)`` with identity labels for each
                gallery item.
            rank: Number of top gallery matches to keep per probe. If ``None``,
                all matches are kept.
            score_class: Do higher scores indicate higher likelyhood of matching
                (``"pos""``, use, e.g., for cosine similarity) or do lower scores
                indicate higher likelyhood of matching (``"neg""``, use, e.g., for
                distances)? Use ``"pos"`` for similarity scores, ``"neg"`` for
                distances.
            equal_class: Do samples with score equal to the threshold get counted as
                a match (``"pos"``) or a non-match (``"neg"``)?

        Returns:
            A :class:`OneToNScores` instance.
        """
        matrix = np.asarray(matrix, dtype=float)  # (P, G)
        probe_labels = np.asarray(probe_labels)  # (P,)
        gallery_labels = np.asarray(gallery_labels)  # (G,)

        # Sort each probe's scores across gallery, best matches first.
        # For score_class="pos": descending (higher is better).
        # For score_class="neg": ascending (lower is better).
        sort_idx = np.argsort(matrix, axis=1)
        if BinaryLabel(score_class) == BinaryLabel.pos:
            sort_idx = sort_idx[:, ::-1]

        # Keep only top-r matches
        if rank is not None:
            sort_idx = sort_idx[:, :rank]
        sorted_matrix = np.take_along_axis(matrix, sort_idx, axis=1)

        # Split into mated (pos) and non-mated (neg) probes
        is_mated = np.isin(probe_labels, gallery_labels)

        scores = OneToNScores(
            pos=sorted_matrix[is_mated],
            neg=sorted_matrix[~is_mated],
            pos_idx=sort_idx[is_mated],
            neg_idx=sort_idx[~is_mated],
            pos_labels=probe_labels[is_mated],
            neg_labels=probe_labels[~is_mated],
            gallery_labels=gallery_labels,
            score_class=score_class,
            equal_class=equal_class,
            is_sorted=True,
        )
        scores._sort_neg()
        return scores

    @staticmethod
    def from_embeddings(
        probe_emb: np.ndarray,
        gallery_emb: np.ndarray,
        probe_labels: np.ndarray,
        gallery_labels: np.ndarray,
        *,
        dist: str = "l2_squared",
        rank: int | None = 1,
        equal_class: BinaryLabel | str = "neg",
        batch_size: int | None = 1e8,
        use_torch: bool = False,
        torch_dtype: "torch.dtype | str" = "float32",
    ):
        """Construct from probe and gallery embeddings.

        Computes pairwise distances using :func:`probe_gallery_distances`, keeps
        the ``rank`` closest gallery items per probe, and splits probes into
        mated and non-mated. The resulting ``score_class`` is always ``"neg"``
        (lower distance is a better match).

        Args:
            probe_emb: Array of shape ``(P, D)`` with probe embeddings.
            gallery_emb: Array of shape ``(G, D)`` with gallery embeddings.
            probe_labels: Array of shape ``(P,)`` with identity labels for each
                probe.
            gallery_labels: Array of shape ``(G,)`` with identity labels for each
                gallery item.
            dist: Distance metric. One of ``"l2_squared"``, ``"l2"`` or
                ``"cosine"``.
            rank: Number of closest gallery items to keep per probe. If ``None``,
                all gallery items are kept.
            equal_class: Do samples with score equal to the threshold get counted as
                a match (``"pos"``) or a non-match (``"neg"``)?
            batch_size: Maximum number of distances to keep in memory at once.
                Controls the probe batch size as ``batch_size // G``. If
                ``None``, all distances are computed at once.
            use_torch: Whether to use PyTorch for distance computation.
            torch_dtype: Data type for PyTorch computations.

        Returns:
            A :class:`OneToNScores` instance.
        """
        probe_labels = np.asarray(probe_labels)
        gallery_labels = np.asarray(gallery_labels)

        distances, indices = probe_gallery_distances(
            probes=probe_emb,
            gallery=gallery_emb,
            dist=dist,
            rank=rank,
            batch_size=batch_size,
            return_indices=True,
            use_torch=use_torch,
            torch_dtype=torch_dtype,
        )

        # Split into mated (pos) and non-mated (neg) probes
        is_mated = np.isin(probe_labels, gallery_labels)

        scores = OneToNScores(
            pos=distances[is_mated],
            neg=distances[~is_mated],
            pos_idx=indices[is_mated],
            neg_idx=indices[~is_mated],
            pos_labels=probe_labels[is_mated],
            neg_labels=probe_labels[~is_mated],
            gallery_labels=gallery_labels,
            score_class="neg",
            equal_class=equal_class,
            is_sorted=True,
        )
        scores._sort_neg()
        return scores

    def fpir(self, threshold: np.ndarray) -> np.ndarray:
        """False Positive Identification Rate for non-mated probes at threshold(s).

        Proportion of non-mated probes whose best gallery match passes the
        threshold, i.e., would be incorrectly accepted as a match.

        Args:
            threshold: Decision threshold(s). Can be a scalar or array.

        Returns:
            FPIR value(s) with the same shape as ``threshold``.
        """
        scores = Scores(
            pos=[],
            neg=self.neg[:, 0],
            score_class=self.score_class,
            equal_class=self.equal_class,
            is_sorted=True,
        )
        return scores.fpr(threshold)

    def fnir(
        self,
        *,
        threshold: np.ndarray | None = None,
        rank: np.ndarray | None = None,
    ) -> np.ndarray:
        """False Negative Identification Rate for mated probes.

        Proportion of mated probes whose true gallery mate is identified, i.e.,
        the mate's score passes the threshold and/or the mate appears within the
        top ``rank`` positions.

        At least one of ``threshold`` or ``rank`` must be provided. When both
        are given, the mate must satisfy both criteria. The output shape is
        ``(*threshold.shape, *rank.shape)``.

        Args:
            threshold: Decision threshold(s). Can be a scalar or array.
            rank: Rank cutoff(s). The mate must appear within the top ``rank``
                positions. Can be a scalar or array.

        Returns:
            Identification rate with shape ``(*threshold.shape, *rank.shape)``.
            Returns a Python scalar when both inputs are scalar or ``None``.
        """
        if threshold is None and rank is None:
            raise ValueError("At least one of threshold or rank must be provided.")

        isscalar = (threshold is None or np.isscalar(threshold)) and (
            rank is None or np.isscalar(rank)
        )
        if threshold is not None:
            threshold = np.asarray(threshold, dtype=float)
        if rank is not None:
            rank = np.asarray(rank)

        Kp = self.pos.shape[0]

        # Truncate to max rank for efficiency
        max_rank = int(np.max(rank)) if rank is not None else self.pos.shape[1]
        pos_idx = self.pos_idx[:, :max_rank]

        # For each mated probe, find the best-ranked mate in the stored results
        gallery_at_idx = self.gallery_labels[pos_idx]  # (Kp, max_rank)
        is_mate = gallery_at_idx == self.pos_labels[:, np.newaxis]  # (Kp, max_rank)
        mate_found = is_mate.any(axis=1)  # (Kp,)
        # argmax returns the first True, i.e., the best-ranked mate
        mate_pos = np.argmax(is_mate, axis=1)  # (Kp,)
        mate_score = self.pos[np.arange(Kp), mate_pos]  # (Kp,)

        # Output shape is (*threshold.shape, *rank.shape) via outer-product
        # broadcasting along the Kp axis.
        t_ndim = threshold.ndim if threshold is not None else 0
        r_ndim = rank.ndim if rank is not None else 0
        extra = t_ndim + r_ndim

        identified = mate_found.reshape((Kp, *([1] * extra)))

        if threshold is not None:
            ms = mate_score.reshape((Kp, *([1] * extra)))
            th = threshold.reshape((*threshold.shape, *([1] * r_ndim)))
            identified = identified & self.threshold_fn(ms, th)

        if rank is not None:
            mp = mate_pos.reshape((Kp, *([1] * extra)))
            rk = rank.reshape((*([1] * t_ndim), *rank.shape))
            identified = identified & (mp < rk)

        result = np.asarray(identified, dtype=float).mean(axis=0)
        if isscalar:
            return result.item()
        return result
