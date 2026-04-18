import numpy as np
import pytest

from score_analysis.one_to_n_scores import OneToNScores


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("score_class", ["pos", "neg"])
def test_from_matrix(rank, score_class):
    scores = OneToNScores.from_matrix(
        matrix=[[2, 1, 0], [1, 3, 2]],
        probes=[0, 10],
        gallery=[10, 11, 12],
        rank=rank,
        score_class=score_class,
    )

    if score_class == "pos":
        expected = OneToNScores(
            pos=np.array([[3, 2, 1]])[:, :rank],
            neg=np.array([[2, 1, 0]])[:, :rank],
            pos_idx=np.array([[11, 12, 10]])[:, :rank],
            neg_idx=np.array([[10, 11, 12]])[:, :rank],
            pos_labels=[10],
            neg_labels=[0],
            gallery_labels=[10, 11, 12],
            score_class=score_class,
        )
    else:
        expected = OneToNScores(
            pos=np.array([[1, 2, 3]])[:, :rank],
            neg=np.array([[0, 1, 2]])[:, :rank],
            pos_idx=np.array([[10, 12, 11]])[:, :rank],
            neg_idx=np.array([[12, 11, 10]])[:, :rank],
            pos_labels=[10],
            neg_labels=[0],
            gallery_labels=[10, 11, 12],
            score_class=score_class,
        )

    assert scores == expected


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
