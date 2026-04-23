from copy import deepcopy

import numpy as np
import pytest

from score_analysis.one_to_n_scores import OneToNScores
from score_analysis.scores import BinaryLabel


def test_equal():
    """Test that equality works as expected."""
    scores1 = OneToNScores(
        neg_rank1=[1, 2, 3],
        pos_rank1=[1, 2],
        pos_mate=[2, 4],
        pos_mate_rank=[1, 3],
        pos_label_rank=[1, 2],
        score_class="neg",
        equal_class="pos",
    )

    scores2 = deepcopy(scores1)
    assert scores1 == scores2

    scores2.pos_label_rank = None
    assert scores1 != scores2

    scores1.pos_label_rank = None
    assert scores1 == scores2


@pytest.mark.parametrize(
    "score_class, equal_class, threshold, expected_fpir",
    [
        ("neg", "pos", 5, 0.4),
        ("neg", "pos", [5, 6], [0.4, 0.6]),  # Vectorised and threshold equals score
        ("neg", "neg", [5, 6], [0.4, 0.4]),  # Change equal_class, same threshold
        ("pos", "pos", [3, 9], [0.8, 0.2]),
    ],
)
def test_fpir_tnir(score_class, equal_class, threshold, expected_fpir):
    scores = OneToNScores(
        neg_rank1=[2, 4, 6, 8, 10],
        pos_rank1=[],
        pos_mate=[],
        pos_mate_rank=[],
        score_class=score_class,
        equal_class=equal_class,
    )
    expected_fpir = np.asarray(expected_fpir)

    fpir = scores.fpir(threshold=threshold)
    np.testing.assert_equal(fpir, expected_fpir)

    tnir = scores.tnir(threshold=threshold)
    np.testing.assert_almost_equal(tnir, 1 - expected_fpir)


@pytest.mark.parametrize(
    "score_class, equal_class, threshold, rank, expected_tpir",
    [
        ("neg", "pos", 10, None, 0.4),
        ("neg", "pos", [10, 12], None, [0.4, 0.6]),  # Vectorised, threshold == score
        ("neg", "neg", [10, 12], None, [0.4, 0.4]),  # Swap equal_class, same threshold
        # ("neg", "neg", None, 2, 0.8),
        # ("neg", "neg", None, [1, 3], [0.4, 1.0]),
        ("neg", "neg", 13, 2, 0.6),
    ],
)
def test_tpir_fnir(score_class, equal_class, threshold, rank, expected_tpir):
    scores = OneToNScores(
        neg_rank1=[],
        pos_rank1=[2, 4, 6, 8, 10],
        pos_mate=[2, 12, 14, 8, 16],
        pos_mate_rank=[1, 2, 3, 1, 2],
        score_class=score_class,
        equal_class=equal_class,
    )
    expected_tpir = np.asarray(expected_tpir)

    tpir = scores.tpir(threshold=threshold, rank=rank)
    np.testing.assert_equal(tpir, expected_tpir)

    fnir = scores.fnir(threshold=threshold, rank=rank)
    np.testing.assert_almost_equal(fnir, 1 - expected_tpir)


@pytest.mark.parametrize(
    "threshold, rank",
    [(0.5, None), ([0.4, 0.6], None), (0.3, 3), (None, 4), (None, [2, 4])],
)
def test_metric_relationships(threshold, rank):
    rng = np.random.default_rng(42)
    scores = OneToNScores(
        neg_rank1=rng.random(20),
        pos_rank1=rng.random(30),
        pos_mate=rng.random(30),
        pos_mate_rank=rng.integers(1, 10, size=30),
        score_class="pos",
        equal_class="pos",
    )

    if threshold is not None:
        fpir = scores.fpir(threshold=threshold)
        tnir = scores.tnir(threshold=threshold)
        np.testing.assert_allclose(fpir, 1.0 - tnir)


@pytest.mark.parametrize(
    "threshold, rank, expected",
    [
        (None, None, 14),
        (7, None, 8),
        ([7], None, [8]),
        ([5, 7], None, [6, 8]),
        (None, 10, 6),
        (None, [10], [6]),
        (None, [10, 13], [6, 8]),
        (13, 20, 10),
        ([13, 13], [20, 20], [[10, 10], [10, 10]]),
    ],
)
def test_mean_rank(threshold, rank, expected):
    scores = OneToNScores(
        neg_rank1=[],
        pos_rank1=[1, 1, 2, 2, 3, 3],
        pos_mate=[2, 4, 6, 16, 10, 12],
        pos_mate_rank=[4, 8, 12, 16, 16, 28],
        score_class="neg",
        equal_class="pos",
    )

    expected = np.asarray(expected)
    mean_rank = scores.mean_rank(threshold=threshold, rank=rank)
    assert np.array_equal(mean_rank, expected)


@pytest.mark.parametrize(
    "score_class, equal_class, threshold, expected",
    [
        ("neg", "pos", 4, 0.6),
        ("neg", "pos", 3, 0.6),
        ("neg", "neg", 3, 0.8),
        ("pos", "pos", 3, 0.2),
        ("pos", "neg", 3, 0.4),
    ],
)
def test_non_match_rate(score_class, equal_class, threshold, expected):
    scores = OneToNScores(
        neg_rank1=[],
        pos_rank1=[1, 3, 5, 7, 9],
        pos_mate=[1, 9, 5, 9, 9],
        pos_mate_rank=[1, 3, 1, 3, 1],
        score_class=score_class,
        equal_class=equal_class,
    )
    expected = np.asarray(expected)

    non_match_rate = scores.non_match_rate(threshold=threshold)
    np.testing.assert_almost_equal(non_match_rate, expected)


@pytest.mark.parametrize(
    "threshold, rank, expected",
    [
        (1.5, None, 0.0),  # Only true matches below threshold
        (2.5, 3, 0.0),  # One false match
        (2.5, 2, 0.2),  # One false match, but above rank threshold
        ([2.5, 3.5], None, [0.0, 0.2]),  # Vectorised threshold
        (None, [2, 3], [0.2, 0.0]),  # Vectorised rank
    ],
)
def test_false_match_rate(threshold, rank, expected):
    """Test that the false match rate is the same as the non-match rate."""
    scores = OneToNScores(
        neg_rank1=[],
        pos_rank1=[1, 2, 3, 4, 5],
        pos_mate=[1, 2.4, 3.8, 9, 5],
        pos_mate_rank=[1, 3, 1, 2, 1],
        score_class="neg",
        equal_class="pos",
    )

    false_match_rate = scores.false_match_rate(threshold=threshold, rank=rank)
    np.testing.assert_almost_equal(false_match_rate, expected)


def test_false_match_rate_shapes():
    scores = OneToNScores(
        neg_rank1=[],
        pos_rank1=[1, 2, 3, 4, 5],
        pos_mate=[1, 2.4, 3.8, 9, 5],
        pos_mate_rank=[1, 3, 1, 2, 1],
        score_class="neg",
        equal_class="pos",
    )

    fmr = scores.false_match_rate(threshold=[1, 2], rank=[1, 2])
    assert fmr.shape == (2, 2)


@pytest.mark.parametrize(
    "rank, expected_dir",
    [(1, 0.4), ([1, 2], [0.4, 0.8]), ([2, 4], [0.8, 1.0])],
)
def test_dir(rank, expected_dir):
    for score_class in ["neg", "pos"]:
        for equal_class in ["neg", "pos"]:
            scores = OneToNScores(
                neg_rank1=[],
                pos_rank1=[2, 4, 6, 8, 10],
                pos_mate=[2, 12, 14, 8, 16],
                pos_mate_rank=[1, 2, 3, 1, 2],
                score_class=score_class,
                equal_class=equal_class,
            )
    expected_dir = np.asarray(expected_dir)

    dir = scores.dir(rank=rank)
    assert np.array_equal(dir, expected_dir)


def test_consolidate():
    """Test that we can consolidate by gallery and probe identity."""
    scores = OneToNScores(
        neg_rank1=[1, 3, 5],
        pos_rank1=[2, 4, 8, 6, 4],
        pos_mate=[4, 6, 8, 10, 12],
        pos_mate_rank=[1, 3, 4, 2, 3],
        pos_label_rank=[1, 2, 3, 2, 1],
        neg_labels=[1, 1, 2],
        pos_labels=[1, 2, 2, 3, 3],
        score_class="neg",
        equal_class="pos",
    )

    consolidated = scores.consolidate(kind="both")
    expected = OneToNScores(
        neg_rank1=[1, 5],
        pos_rank1=[2, 4, 6],
        pos_mate=[4, 6, 10],
        pos_mate_rank=[1, 2, 2],
        pos_label_rank=[1, 2, 2],
        neg_labels=[1, 2],
        pos_labels=[1, 2, 3],
        score_class="neg",
        equal_class="pos",
    )
    assert consolidated == expected

    scores.score_class = BinaryLabel.pos
    consolidated = scores.consolidate(kind="both")
    expected = OneToNScores(
        neg_rank1=[3, 5],
        pos_rank1=[2, 8, 4],
        pos_mate=[4, 8, 12],
        pos_mate_rank=[1, 3, 1],
        pos_label_rank=[1, 3, 1],
        neg_labels=[1, 2],
        pos_labels=[1, 2, 3],
        score_class="pos",
        equal_class="pos",
    )
    assert consolidated == expected


def test_rank_one_scores():
    """Test that we can retrieve rank one scores for all subsets."""
    scores = OneToNScores(
        neg_rank1=[1, 2, 3],
        pos_rank1=[1, 2],
        pos_mate=[2, 4],
        pos_mate_rank=[1, 3],
        score_class="neg",
        equal_class="pos",
    )

    rank1 = scores.rank_one_scores(subset="mated")
    assert np.array_equal(rank1, [1, 2])

    rank1 = scores.rank_one_scores(subset="non_mated")
    assert np.array_equal(rank1, [1, 2, 3])

    rank1 = scores.rank_one_scores(subset="all")
    rank1.sort()
    assert np.array_equal(rank1, [1, 1, 2, 2, 3])


def test_to_binary_scores():
    """Test conversion to binary scores with and without rank truncation."""
    scores = OneToNScores(
        neg_rank1=[1, 2, 3],
        pos_rank1=[1, 2],
        pos_mate=[2, 4],
        pos_mate_rank=[1, 3],
        score_class="neg",
        equal_class="pos",
    )

    binary_scores = scores.to_binary_scores(rank=None)
    assert np.array_equal(binary_scores.neg, [1, 2, 3])
    assert np.array_equal(binary_scores.pos, [2, 4])

    # When we truncate by rank, we need to add +/-inf in the right place
    binary_scores = scores.to_binary_scores(rank=1)
    assert np.array_equal(binary_scores.neg, [1, 2, 3])
    assert np.array_equal(binary_scores.pos, [2, np.inf])

    scores.score_class = "pos"
    binary_scores = scores.to_binary_scores(rank=1)
    assert np.array_equal(binary_scores.neg, [1, 2, 3])
    assert np.array_equal(binary_scores.pos, [-np.inf, 2])


def test_raises_errors():
    scores = OneToNScores(
        neg_rank1=[1, 2, 3],
        pos_rank1=[1, 2],
        pos_mate=[2, 4],
        pos_mate_rank=[1, 3],
        pos_label_rank=[1, 2],
        neg_labels=[1, 1, 2],
        pos_labels=[1, 3],
        score_class="neg",
        equal_class="pos",
    )

    with pytest.raises(ValueError):
        scores.tpir(threshold=None, rank=None)
    with pytest.raises(ValueError):
        scores.rank_one_scores(subset="invalid")

    # Consolidation
    with pytest.raises(ValueError):
        scores.consolidate(kind="invalid")

    scores.pos_label_rank = None
    with pytest.raises(ValueError):
        scores.consolidate(kind="gallery")

    scores.pos_labels = None
    with pytest.raises(ValueError):
        scores.consolidate(kind="probe")

    scores.pos_labels = np.asarray([1, 3])
    scores.neg_labels = None
    with pytest.raises(ValueError):
        scores.consolidate(kind="probe")
