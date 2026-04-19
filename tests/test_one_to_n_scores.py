import numpy as np
import pytest

from score_analysis.one_to_n_scores import OneToNScores


@pytest.mark.parametrize("rank", [None, 1, 2, 3])
@pytest.mark.parametrize("score_class", ["pos", "neg"])
def test_from_matrix(rank, score_class):
    scores = OneToNScores.from_matrix(
        matrix=[[2, 1, 0], [1, 3, 2]],
        probe_labels=[0, 10],
        gallery_labels=[10, 11, 12],
        rank=rank,
        score_class=score_class,
    )

    slice_ = slice(None) if rank is None else slice(rank)
    if score_class == "pos":
        expected = OneToNScores(
            pos=np.array([[3, 2, 1]])[:, slice_],
            neg=np.array([[2, 1, 0]])[:, slice_],
            pos_idx=np.array([[11, 12, 10]])[:, slice_],
            neg_idx=np.array([[10, 11, 12]])[:, slice_],
            pos_labels=[10],
            neg_labels=[0],
            gallery_labels=[10, 11, 12],
            score_class=score_class,
        )
    else:
        expected = OneToNScores(
            pos=np.array([[1, 2, 3]])[:, slice_],
            neg=np.array([[0, 1, 2]])[:, slice_],
            pos_idx=np.array([[10, 12, 11]])[:, slice_],
            neg_idx=np.array([[12, 11, 10]])[:, slice_],
            pos_labels=[10],
            neg_labels=[0],
            gallery_labels=[10, 11, 12],
            score_class=score_class,
        )

    assert scores == expected


@pytest.mark.parametrize("rank", [None, 1, 2, 3])
def test_from_embeddings(rank):
    """Test that from_matrix and from_embeddings give the same result."""
    probe_labels = [0, 3]
    gallery_labels = [2, 1, 0]
    probe_emb = [[0], [6]]
    gallery_emb = [[3], [4], [5]]
    matrix = [[3, 4, 5], [3, 2, 1]]

    scores_from_matrix = OneToNScores.from_matrix(
        matrix=matrix,
        probe_labels=probe_labels,
        gallery_labels=gallery_labels,
        rank=rank,
        score_class="neg",
        equal_class="neg",
    )

    scores_from_emb = OneToNScores.from_embeddings(
        probe_emb=probe_emb,
        gallery_emb=gallery_emb,
        probe_labels=probe_labels,
        gallery_labels=gallery_labels,
        dist="l2",
        rank=rank,
        equal_class="neg",
    )

    assert scores_from_matrix == scores_from_emb


# def test_fpir(score_class, equal_class):
#     scores = OneToNScores(
#         pos=[],
#         neg=[0, 1, 1, 1, 2, 3, 4, 5, 6, 6],
#         pos_idx=[],
#         neg_idx=[],
#         probes=[0, 1, 2, 3, 4],
#         gallery=[10, 11, 12, 13, 14],
#         score_class=score_class,
#         equal_class=equal_class,
#     )


#     dist_matrix = [
#         [0, 1, 2, 3, 4],
#         [1, 1, 2, 3, 4],
#         []
#     ]
