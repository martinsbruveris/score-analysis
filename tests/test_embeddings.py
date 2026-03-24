import numpy as np
import pytest

from score_analysis.embeddings import embedding_distances


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
