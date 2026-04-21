import numpy as np
import pytest
import torch

from score_analysis.embeddings import (
    _get_torch_dtype,
    cross_embedding_distances,
    embedding_distances,
)


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
    print(pos_idx)
    print(neg_idx)
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
