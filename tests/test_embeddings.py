import warnings

import numpy as np
import pytest
import torch

from score_analysis.embeddings import (
    _get_torch_dtype,
    cross_embedding_distances,
    embedding_distances,
    probe_gallery_distances,
    # probe_gallery_distances_torch,
)
from score_analysis.one_to_n_scores import OneToNScores


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances(use_torch):
    """Test basic embedding distance calculations."""
    kwargs = {"batch_size": None, "use_torch": use_torch}

    emb = np.array([[1], [2], [3], [4], [5]])
    labels = np.array([0, 0, 1, 1, 1])

    scores = embedding_distances(emb, labels, dist="l2", **kwargs)
    assert np.array_equal(scores.pos, [1, 1, 1, 2])
    assert np.array_equal(scores.neg, [1, 2, 2, 3, 3, 4])

    scores = embedding_distances(emb, labels, dist="l2_squared", **kwargs)
    assert np.array_equal(scores.pos, [1, 1, 1, 4])
    assert np.array_equal(scores.neg, [1, 4, 4, 9, 9, 16])

    scores = embedding_distances(emb, labels, dist="cosine", **kwargs)
    assert np.array_equal(scores.pos, [0, 0, 0, 0])
    assert np.array_equal(scores.neg, [0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_limits(use_torch):
    """
    Test embedding distance calculations with limits on positive and negative pairs.
    """
    kwargs = {"dist": "l2", "batch_size": 8, "use_torch": use_torch}

    emb = np.array([[0], [1], [2], [3], [4]])
    labels = np.array([0, 0, 1, 1, 1])

    scores = embedding_distances(emb, labels, pos_limit=2, neg_limit=3, **kwargs)
    assert np.array_equal(scores.pos, [1, 2])
    assert np.array_equal(scores.neg, [1, 2, 2])
    assert scores.nb_hard_pos == 2
    assert scores.nb_hard_neg == 3

    scores = embedding_distances(emb, labels, pos_limit=0.5, neg_limit=0.5, **kwargs)
    assert np.array_equal(scores.pos, [1, 2])
    assert np.array_equal(scores.neg, [1, 2, 2])
    assert scores.nb_hard_pos == 2
    assert scores.nb_hard_neg == 3


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_invalid_distance(use_torch):
    """Test that an invalid distance metric raises an error."""
    emb = np.array([[1], [2], [3]])
    labels = np.array([0, 0, 1])

    with pytest.raises(ValueError):
        embedding_distances(emb, labels, dist="invalid_distance", use_torch=use_torch)


@pytest.mark.parametrize("use_torch", [False, True])
def test_single_embedding(use_torch):
    """Only one embedding produces no scores."""
    emb = np.array([[1, 2, 3]])
    labels = np.array([0])

    scores = embedding_distances(emb, labels, use_torch=use_torch)
    assert len(scores.pos) == 0
    assert len(scores.neg) == 0


@pytest.mark.parametrize("use_torch", [False, True])
def test_single_unique_label(use_torch):
    """All embeddings share the same label -> no negative scores."""
    emb = np.array([[1], [2], [3]])
    labels = np.array([0, 0, 0])

    scores = embedding_distances(emb, labels, use_torch=use_torch)
    assert len(scores.pos) == 3  # C(3,2) = 3 positive pairs
    assert len(scores.neg) == 0


@pytest.mark.parametrize("use_torch", [False, True])
def test_all_unique_labels(use_torch):
    """Every embedding has a unique label -> no positive scores."""
    emb = np.array([[1], [2], [3]])
    labels = np.array([0, 1, 2])

    scores = embedding_distances(emb, labels, use_torch=use_torch)
    assert len(scores.pos) == 0
    assert len(scores.neg) == 3  # C(3,2) = 3 negative pairs


@pytest.mark.parametrize("dist", ["l2", "l2_squared", "cosine"])
def test_torch_numpy_equality(dist):
    """Results from use_torch=True and use_torch=False should match."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    labels = rng.integers(0, 4, size=20)

    scores_np = embedding_distances(emb, labels, dist=dist, use_torch=False)
    scores_torch = embedding_distances(emb, labels, dist=dist, use_torch=True)

    np.testing.assert_allclose(scores_np.pos, scores_torch.pos, rtol=1e-6)
    np.testing.assert_allclose(scores_np.neg, scores_torch.neg, rtol=1e-6)


def test_torch_dtype():
    """Test that the use_torch option respects the dtype of the input embeddings."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((10, 4)).astype(np.float64)
    labels = rng.integers(0, 2, size=10)

    scores = embedding_distances(emb, labels, use_torch=True, torch_dtype="float32")
    assert scores.pos.dtype == np.float64
    assert scores.neg.dtype == np.float64


def test_get_torch_dtype():
    """Test the _get_torch_dtype function."""
    assert _get_torch_dtype(None) is None
    assert _get_torch_dtype("float32") == torch.float32
    assert _get_torch_dtype("float64") == torch.float64
    assert _get_torch_dtype(torch.float16) == torch.float16
    with pytest.raises(TypeError):
        _get_torch_dtype(3)


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_non_writeable_input(use_torch):
    """Non-writeable arrays should not trigger warnings."""
    emb = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    emb.flags.writeable = False
    labels = np.array([0, 0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = embedding_distances(emb, labels, use_torch=use_torch)

    assert len(scores.pos) == 1
    assert len(scores.neg) == 2


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_indices(use_torch):
    """Test that we return the correct indices for positive and negative pairs."""
    emb = np.array([[0], [1], [3], [7], [14]])
    labels = np.array([0, 0, 1, 1, 1])

    scores, pos_idx, neg_idx = embedding_distances(
        emb=emb,
        labels=labels,
        dist="l2",
        batch_size=None,
        use_torch=use_torch,
        return_indices=True,
    )
    assert np.array_equal(scores.pos, [1, 4, 7, 11])
    assert np.array_equal(scores.neg, [2, 3, 6, 7, 13, 14])
    assert np.array_equal(pos_idx, [[0, 1], [2, 3], [3, 4], [2, 4]])
    assert np.array_equal(neg_idx, [[1, 2], [0, 2], [1, 3], [0, 3], [1, 4], [0, 4]])


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_indices_limits(use_torch):
    """Test that we return the correct indices for positive and negative pairs."""
    emb = np.array([[0], [1], [3], [7], [14]])
    labels = np.array([0, 0, 1, 1, 1])

    scores, pos_idx, neg_idx = embedding_distances(
        emb=emb,
        labels=labels,
        dist="l2",
        pos_limit=2,
        neg_limit=2,
        batch_size=None,
        use_torch=use_torch,
        return_indices=True,
    )
    assert np.array_equal(scores.pos, [7, 11])
    assert np.array_equal(scores.neg, [2, 3])
    assert np.array_equal(pos_idx, [[3, 4], [2, 4]])
    assert np.array_equal(neg_idx, [[1, 2], [0, 2]])


def _reference_hard_pairs(emb, labels, dist, pos_limit=None, neg_limit=None):
    """Brute-force reference: all pairs, then keep the hardest ones.

    Hardest positives are the ones with the largest distances, hardest negatives the
    ones with the smallest distances. Returns sorted arrays.
    """
    emb = np.asarray(emb, dtype=np.float64)
    labels = np.asarray(labels)
    if dist == "l2_squared":
        d = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=-1)
    elif dist == "l2":
        d = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=-1))
    elif dist == "cosine":
        normed = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10, None)
        d = 1.0 - normed @ normed.T
    else:
        raise ValueError(dist)

    i, j = np.triu_indices(len(emb), k=1)
    same = labels[i] == labels[j]
    pos = np.sort(d[i, j][same])
    neg = np.sort(d[i, j][~same])
    if pos_limit is not None:
        pos = pos[len(pos) - min(pos_limit, len(pos)) :]
    if neg_limit is not None:
        neg = neg[: min(neg_limit, len(neg))]
    return pos, neg


def _random_embeddings(n, dim, nb_labels, seed):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    labels = rng.integers(0, nb_labels, size=n)
    return emb, labels


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("dist", ["l2", "l2_squared", "cosine"])
@pytest.mark.parametrize("batch_size", [None, 64, 4096])
def test_embedding_distances_limits_are_exact(use_torch, dist, batch_size):
    """Limited results must be exactly the hardest pairs, over several batchings.

    Batching matters here: the hardest pairs are tracked incrementally across row
    blocks, and candidates that cannot make the cut are discarded before being
    materialized. This must not change the result.
    """
    emb, labels = _random_embeddings(n=60, dim=5, nb_labels=8, seed=11)
    pos_limit, neg_limit = 7, 13

    scores = embedding_distances(
        emb,
        labels,
        dist=dist,
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        batch_size=batch_size,
        use_torch=use_torch,
    )
    pos_ref, neg_ref = _reference_hard_pairs(emb, labels, dist, pos_limit, neg_limit)

    np.testing.assert_allclose(np.sort(scores.pos), pos_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.sort(scores.neg), neg_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("limit", list(range(1, 12)))
def test_embedding_distances_limits_sweep(use_torch, limit):
    """Sweep the limit, so that every buffer-vs-limit alignment is exercised.

    In particular this covers the case where a buffer sits exactly at the limit
    without having been trimmed, which is the boundary of the pre-filter.
    """
    # Integer embeddings, so that l2_squared distances are exactly representable
    # and the reference comparison can be exact.
    emb = np.array([[0], [1], [3], [7], [14], [15], [22], [30]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])

    scores = embedding_distances(
        emb,
        labels,
        dist="l2_squared",
        pos_limit=limit,
        neg_limit=limit,
        batch_size=8,  # forces several row blocks
        use_torch=use_torch,
    )
    pos_ref, neg_ref = _reference_hard_pairs(emb, labels, "l2_squared", limit, limit)

    assert np.array_equal(np.sort(scores.pos), pos_ref)
    assert np.array_equal(np.sort(scores.neg), neg_ref)


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_limits_with_ties(use_torch):
    """Ties at the cut-off must not change the returned distances.

    Many pairs share the same distance here, so the hardest-K set is not unique.
    Which pair is picked may vary, but the multiset of distances may not.
    """
    rng = np.random.default_rng(3)
    # Only a handful of distinct embeddings, repeated -> heavy ties in the distances
    distinct = np.arange(5, dtype=np.float32)[:, None]
    emb = distinct[rng.integers(0, len(distinct), size=40)]
    labels = rng.integers(0, 4, size=40)

    for limit in [1, 3, 17, 40]:
        scores = embedding_distances(
            emb,
            labels,
            dist="l2_squared",
            pos_limit=limit,
            neg_limit=limit,
            batch_size=16,
            use_torch=use_torch,
        )
        pos_ref, neg_ref = _reference_hard_pairs(
            emb, labels, "l2_squared", limit, limit
        )
        assert np.array_equal(np.sort(scores.pos), pos_ref)
        assert np.array_equal(np.sort(scores.neg), neg_ref)


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("limit", [None, 10**6])
def test_embedding_distances_limits_never_binding(use_torch, limit):
    """A limit larger than the number of pairs must return all pairs unchanged."""
    emb, labels = _random_embeddings(n=40, dim=4, nb_labels=6, seed=5)

    kwargs = {"dist": "l2", "batch_size": 64, "use_torch": use_torch}
    unlimited = embedding_distances(emb, labels, **kwargs)
    limited = embedding_distances(
        emb, labels, pos_limit=limit, neg_limit=limit, **kwargs
    )

    assert np.array_equal(np.sort(unlimited.pos), np.sort(limited.pos))
    assert np.array_equal(np.sort(unlimited.neg), np.sort(limited.neg))
    assert limited.nb_easy_pos == 0
    assert limited.nb_easy_neg == 0


@pytest.mark.parametrize("use_torch", [False, True])
def test_embedding_distances_limits_indices_are_consistent(use_torch):
    """Returned indices must point at pairs that really have the returned distances.

    Candidates are dropped by narrowing the mask that gathers both the distances and
    the indices, so the two must stay aligned. Ties may change which of several equal
    pairs is reported, hence the check is that the reported pair reproduces the
    reported distance, not that the indices match a fixed list.
    """
    emb, labels = _random_embeddings(n=50, dim=4, nb_labels=7, seed=17)
    pos_limit, neg_limit = 6, 9

    scores, pos_idx, neg_idx = embedding_distances(
        emb,
        labels,
        dist="l2_squared",
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        batch_size=64,
        use_torch=use_torch,
        return_indices=True,
    )

    full = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=-1)
    for idx, dists, expect_same_label in [
        (pos_idx, scores.pos, True),
        (neg_idx, scores.neg, False),
    ]:
        assert len(idx) == len(dists)
        rows, cols = idx[:, 0], idx[:, 1]
        assert np.all(rows < cols)  # upper triangle only
        np.testing.assert_allclose(full[rows, cols], dists, rtol=1e-5, atol=1e-6)
        assert np.all((labels[rows] == labels[cols]) == expect_same_label)


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("batch_size", [None, 64])
def test_cross_embedding_distances_limits_are_exact(use_torch, batch_size):
    """The same exactness requirement for the two-sided variant."""
    emb_a, labels_a = _random_embeddings(n=30, dim=4, nb_labels=5, seed=21)
    emb_b, labels_b = _random_embeddings(n=25, dim=4, nb_labels=5, seed=22)
    pos_limit, neg_limit = 5, 11

    scores = cross_embedding_distances(
        emb_a,
        emb_b,
        labels_a,
        labels_b,
        dist="l2_squared",
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        batch_size=batch_size,
        use_torch=use_torch,
    )

    full = ((emb_a[:, None, :] - emb_b[None, :, :]) ** 2).sum(axis=-1).ravel()
    same = (labels_a[:, None] == labels_b[None, :]).ravel()
    pos_ref = np.sort(full[same])[-pos_limit:]
    neg_ref = np.sort(full[~same])[:neg_limit]

    np.testing.assert_allclose(np.sort(scores.pos), pos_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.sort(scores.neg), neg_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_torch", [False, True])
def test_cross_embedding_distances(use_torch):
    """Test basic embedding distance calculations."""
    kwargs = {"batch_size": None, "use_torch": use_torch}

    emb_a = np.array([[1], [2], [3], [4], [5]])
    emb_b = np.array([[2], [3], [6]])
    labels_a = np.array([0, 0, 1, 1, 1])
    labels_b = np.array([0, 0, 1])

    scores = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, dist="l2", **kwargs
    )
    assert np.array_equal(scores.pos, [0, 1, 1, 1, 2, 2, 3])
    assert np.array_equal(scores.neg, [0, 1, 1, 2, 2, 3, 4, 5])

    scores = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, dist="l2_squared", **kwargs
    )
    # We get some numerical errors here at the 3e-6 level...
    np.testing.assert_allclose(scores.pos, [0, 1, 1, 1, 4, 4, 9], atol=1e-5)
    np.testing.assert_allclose(scores.neg, [0, 1, 1, 4, 4, 9, 16, 25], atol=1e-5)

    scores = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, dist="cosine", **kwargs
    )
    assert np.array_equal(scores.pos, [0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(scores.neg, [0, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("use_torch", [False, True])
def test_cross_embedding_distances_limits(use_torch):
    """
    Test embedding distance calculations with limits on positive and negative pairs.
    """
    kwargs = {"dist": "l2", "batch_size": 8, "use_torch": use_torch}

    emb_a = np.array([[1], [2], [3], [4], [5]])
    emb_b = np.array([[2], [3], [6], [7]])
    labels_a = np.array([0, 0, 1, 2, 2])
    labels_b = np.array([0, 0, 0, 2])

    scores = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, pos_limit=2, neg_limit=3, **kwargs
    )
    assert np.array_equal(scores.pos, [4, 5])
    assert np.array_equal(scores.neg, [0, 1, 1])
    assert scores.nb_hard_pos == 2
    assert scores.nb_hard_neg == 3

    scores = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, pos_limit=0.5, neg_limit=0.5, **kwargs
    )
    assert np.array_equal(scores.pos, [2, 3, 4, 5])
    assert np.array_equal(scores.neg, [0, 1, 1, 1, 2, 2])
    assert scores.nb_hard_pos == 4
    assert scores.nb_hard_neg == 6


@pytest.mark.parametrize("use_torch", [False, True])
def test_cross_embedding_invalid_distance(use_torch):
    """Test that an invalid distance metric raises an error."""
    emb = np.array([[1], [2], [3]])
    labels = np.array([0, 0, 1])

    with pytest.raises(ValueError):
        cross_embedding_distances(
            emb, emb, labels, labels, dist="invalid_distance", use_torch=use_torch
        )


@pytest.mark.parametrize("use_torch", [False, True])
def test_cross_embedding_single_embedding(use_torch):
    """Only one embedding produces no positive/negative scores depending on label."""
    emb = np.array([[1, 2, 3]])

    scores = cross_embedding_distances(emb, emb, [0], [0], use_torch=use_torch)
    assert len(scores.pos) == 1
    assert len(scores.neg) == 0

    scores = cross_embedding_distances(emb, emb, [0], [1], use_torch=use_torch)
    assert len(scores.pos) == 0
    assert len(scores.neg) == 1


@pytest.mark.parametrize("dist", ["l2", "l2_squared", "cosine"])
def test_cross_distance_torch_numpy_equality(dist):
    """Results from use_torch=True and use_torch=False should match."""
    rng = np.random.default_rng(42)
    emb_a = rng.standard_normal((20, 8)).astype(np.float32)
    emb_b = rng.standard_normal((14, 8)).astype(np.float32)
    labels_a = rng.integers(0, 4, size=20)
    labels_b = rng.integers(0, 6, size=14)

    scores_np = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, dist=dist, use_torch=False
    )
    scores_torch = cross_embedding_distances(
        emb_a, emb_b, labels_a, labels_b, dist=dist, use_torch=True
    )

    np.testing.assert_allclose(scores_np.pos, scores_torch.pos, rtol=1e-6)
    np.testing.assert_allclose(scores_np.neg, scores_torch.neg, rtol=1e-6)


def test_cross_distance_torch_dtype():
    """Test that the use_torch option respects the dtype of the input embeddings."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((10, 4)).astype(np.float64)
    labels = rng.integers(0, 2, size=10)

    scores = cross_embedding_distances(
        emb, emb, labels, labels, use_torch=True, torch_dtype="float32"
    )
    assert scores.pos.dtype == np.float64
    assert scores.neg.dtype == np.float64


@pytest.mark.parametrize("use_torch", [False, True])
def test_cross_embedding_distances_indices(use_torch):
    """Test that we return the correct indices for positive and negative pairs."""
    emb_a = np.array([[0], [1], [2]])
    emb_b = np.array([[4], [6], [9]])
    labels_a = np.array([0, 0, 1])
    labels_b = np.array([0, 0, 1])

    scores, pos_idx, neg_idx = cross_embedding_distances(
        emb_a=emb_a,
        emb_b=emb_b,
        labels_a=labels_a,
        labels_b=labels_b,
        dist="l2",
        batch_size=None,
        use_torch=use_torch,
        return_indices=True,
    )
    assert np.array_equal(scores.pos, [3, 4, 5, 6, 7])
    assert np.array_equal(scores.neg, [2, 4, 8, 9])
    assert np.array_equal(pos_idx, [[1, 0], [0, 0], [1, 1], [0, 1], [2, 2]])
    assert np.array_equal(neg_idx, [[2, 0], [2, 1], [1, 2], [0, 2]])

    scores, pos_idx, neg_idx = cross_embedding_distances(
        emb_a=emb_a,
        emb_b=emb_b,
        labels_a=labels_a,
        labels_b=labels_b,
        dist="l2",
        pos_limit=3,
        neg_limit=3,
        batch_size=None,
        use_torch=use_torch,
        return_indices=True,
    )
    assert np.array_equal(scores.pos, [5, 6, 7])
    assert np.array_equal(scores.neg, [2, 4, 8])
    assert np.array_equal(pos_idx, [[1, 1], [0, 1], [2, 2]])
    assert np.array_equal(neg_idx, [[2, 0], [2, 1], [1, 2]])


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("batch_size", [None, 4])
def test_probe_gallery_distances(use_torch, batch_size):
    """Test basic probe-gallery distance calculations."""
    kwargs = {"batch_size": batch_size, "use_torch": use_torch}

    probe = [[0], [1], [2]]
    gallery = [[5], [4], [3]]
    probe_labels = [4, 5, 1]
    gallery_labels = [1, 2, 2]

    scores = probe_gallery_distances(
        probe=probe,
        gallery=gallery,
        probe_labels=probe_labels,
        gallery_labels=gallery_labels,
        dist="l2",
        **kwargs,
    )
    expected = OneToNScores(
        neg_rank1=[2, 3],
        pos_rank1=[1],
        pos_mate=[3],
        pos_mate_rank=[3],
        pos_label_rank=[2],
        neg_labels=[5, 4],
        pos_labels=[1],
        score_class="neg",
        equal_class="pos",
    )
    assert scores == expected


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("batch_size", [None, 4])
def test_probe_gallery_distances_return_indices(use_torch, batch_size):
    """Test basic probe-gallery distance calculations."""
    kwargs = {"batch_size": batch_size, "use_torch": use_torch}

    probe = [[0.0], [1.0], [2.0]]
    gallery = [[5.0], [4.0], [3.0]]
    probe_labels = [4, 5, 1]
    gallery_labels = [1, 2, 2]

    _, indices = probe_gallery_distances(
        probe=probe,
        gallery=gallery,
        probe_labels=probe_labels,
        gallery_labels=gallery_labels,
        dist="l2",
        return_indices=True,
        **kwargs,
    )
    assert np.array_equal(indices.neg_rank1, [[1, 2], [0, 2]])
    assert np.array_equal(indices.pos_rank1, [[2, 2]])
    assert np.array_equal(indices.pos_mate, [[2, 0]])


@pytest.mark.parametrize("use_torch", [False, True])
def test_probe_gallery_invalid_distance(use_torch):
    """Test that an invalid distance metric raises an error."""
    with pytest.raises(ValueError):
        probe_gallery_distances(
            probe=[[0], [1], [2]],
            gallery=[[5], [4], [3]],
            probe_labels=[4, 5, 1],
            gallery_labels=[1, 2, 2],
            dist="invalid_distance",
            use_torch=use_torch,
        )


@pytest.mark.parametrize("dist", ["l2", "l2_squared", "cosine"])
def test_probe_gallery_torch_numpy_equality(dist):
    """Results from use_torch=True and use_torch=False should match."""
    rng = np.random.default_rng(42)
    probe = rng.standard_normal((20, 8)).astype(np.float32)
    gallery = rng.standard_normal((30, 8)).astype(np.float32)
    probe_labels = rng.integers(0, 4, size=20)
    gallery_labels = rng.integers(0, 4, size=30)

    args = (probe, gallery, probe_labels, gallery_labels)
    kwargs = {"dist": dist, "return_indices": True}

    scores_np, indices_np = probe_gallery_distances(*args, **kwargs, use_torch=False)
    scores_torch, indices_torch = probe_gallery_distances(
        *args, **kwargs, use_torch=True
    )

    np.testing.assert_allclose(scores_np.neg_rank1, scores_torch.neg_rank1, rtol=1e-6)
    np.testing.assert_allclose(scores_np.pos_rank1, scores_torch.pos_rank1, rtol=1e-6)
    np.testing.assert_allclose(scores_np.pos_mate, scores_torch.pos_mate, rtol=1e-6)
    np.testing.assert_equal(scores_np.pos_mate_rank, scores_torch.pos_mate_rank)
    np.testing.assert_equal(scores_np.pos_label_rank, scores_torch.pos_label_rank)
    np.testing.assert_equal(scores_np.neg_labels, scores_torch.neg_labels)
    np.testing.assert_equal(scores_np.pos_labels, scores_torch.pos_labels)

    np.testing.assert_equal(indices_np.neg_rank1, indices_torch.neg_rank1)
    np.testing.assert_equal(indices_np.pos_rank1, indices_torch.pos_rank1)
    np.testing.assert_equal(indices_np.pos_mate, indices_torch.pos_mate)


@pytest.mark.parametrize("use_torch", [False, True])
@pytest.mark.parametrize("mated", [True, False])
def test_probe_gallery_only_mated_or_non_mated(use_torch, mated):
    """Test probe-gallery distances when only mated or non-mated probes are present."""

    probe_gallery_distances(
        probe=[[0], [1], [2], [3]],
        gallery=[[5], [4], [3]],
        probe_labels=[0, 1, 2, 3],
        gallery_labels=[0, 1, 2] if mated else [4, 5, 6],
        batch_size=2,
        use_torch=use_torch,
    )
