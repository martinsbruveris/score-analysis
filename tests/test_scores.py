import numpy as np
import pytest

from score_analysis.scores import Scores


@pytest.mark.parametrize(
    "pos, neg, threshold, score_class, equal_class, expected",
    [
        # Normal case, all scores and threshold are unique
        [[0, 1, 2, 3], [0], 2.5, "pos", "pos", [[1, 3], [0, 1]]],
        # No positive scores
        [[], [0], 1.5, "pos", "pos", [[0, 0], [0, 1]]],
        # No negative scores
        [[0], [], 1.5, "pos", "pos", [[0, 1], [0, 0]]],
        # Count threshold to pos
        [[0, 1, 1, 2, 3], [1, 1, 2], 1, "pos", "pos", [[4, 1], [3, 0]]],
        [[1, 1, 2], [0, 1, 1, 2, 3], 1, "pos", "pos", [[3, 0], [4, 1]]],
        # Count threshold to neg
        [[0, 1, 1, 2, 3], [1, 1, 2], 1, "pos", "neg", [[2, 3], [1, 2]]],
        [[1, 1, 2], [0, 1, 1, 2, 3], 1, "pos", "neg", [[1, 2], [2, 3]]],
        # Reverse score direction
        [[0, 1, 2, 3], [0], 2.5, "neg", "pos", [[3, 1], [1, 0]]],
        # Reverse score direction with threshold equal to scores, count to pos
        [[0, 1, 1, 2, 3], [1, 1, 2], 1, "neg", "pos", [[3, 2], [2, 1]]],
        [[1, 1, 2], [0, 1, 1, 2, 3], 1, "neg", "pos", [[2, 1], [3, 2]]],
        # Reverse score direction with threshold equal to scores, count to neg
        [[0, 1, 1, 2, 3], [1, 1, 2], 1, "neg", "neg", [[1, 4], [0, 3]]],
        [[1, 1, 2], [0, 1, 1, 2, 3], 1, "neg", "neg", [[0, 3], [1, 4]]],
        # Threshold lower than all scores
        [[0, 1, 2], [0, 1, 2], -1, "pos", "pos", [[3, 0], [3, 0]]],
        [[0, 1, 2], [0, 1, 2], -1, "neg", "pos", [[0, 3], [0, 3]]],
        # Threshold higher than all scores
        [[0, 1, 2], [0, 1, 2], 3, "pos", "pos", [[0, 3], [0, 3]]],
        [[0, 1, 2], [0, 1, 2], 3, "neg", "pos", [[3, 0], [3, 0]]],
        # Threshold at lower extreme value
        [[0, 1, 2], [0, 1, 2], 0, "pos", "pos", [[3, 0], [3, 0]]],
        [[0, 1, 2], [0, 1, 2], 0, "pos", "neg", [[2, 1], [2, 1]]],
        [[0, 1, 2], [0, 1, 2], 0, "neg", "pos", [[1, 2], [1, 2]]],
        [[0, 1, 2], [0, 1, 2], 0, "neg", "neg", [[0, 3], [0, 3]]],
        # Threshold at higher extreme value
        [[0, 1, 2], [0, 1, 2], 2, "pos", "pos", [[1, 2], [1, 2]]],
        [[0, 1, 2], [0, 1, 2], 2, "pos", "neg", [[0, 3], [0, 3]]],
        [[0, 1, 2], [0, 1, 2], 2, "neg", "pos", [[3, 0], [3, 0]]],
        [[0, 1, 2], [0, 1, 2], 2, "neg", "neg", [[2, 1], [2, 1]]],
        # Vectorized version
        [[0, 1, 2, 3], [0], [[2.5]], "pos", "pos", [[[[1, 3], [0, 1]]]]],
    ],
)
def test_cm_at_threshold(pos, neg, threshold, score_class, equal_class, expected):
    scores = Scores(pos, neg, score_class=score_class, equal_class=equal_class)
    cm = scores.cm_at_threshold(threshold)
    np.testing.assert_equal(cm.matrix, expected)


@pytest.mark.parametrize(
    "pos, neg, threshold, expected",
    [
        # Simple scalar version
        [[0, 1, 2, 3], [0, 1, 2], 1.5, 0.5],
        # Vectorized version
        [[0, 1, 2, 3], [], [-1, 0, 1.5, 3, 4], [1., 1., 0.5, 0.25, 0.]],
    ],
)
def test_tpr_at_threshold(pos, neg, threshold, expected):
    scores = Scores(pos, neg)
    tpr = scores.tpr_at_threshold(threshold)
    np.testing.assert_equal(tpr, expected)


@pytest.mark.parametrize(
    "pos, tpr, score_class, equal_class, expected",
    [
        # First example
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "pos", "pos", [4, 4, 3, 2, 1]],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "pos", "pos", [4, 3.5, 2.5, 1.5]],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "pos", "neg", [4, 3, 2, 1, 1]],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "pos", "neg", [3.5, 2.5, 1.5, 1]],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "neg", "pos", [1, 1, 2, 3, 4]],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "neg", "pos", [1, 1.5, 2.5, 3.5]],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "neg", "neg", [1, 2, 3, 4, 4]],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "neg", "neg", [1.5, 2.5, 3.5, 4]],
        # Simple interpolation
        [[0, 1], 0.6, "pos", "pos", 0.8],
        [[0, 1], 0.6, "pos", "neg", 0],
        [[0, 1], 0.6, "neg", "pos", 0.2],
        [[0, 1], 0.6, "neg", "neg", 1],
        # Edge cases
        [[0, 1], -0.1, "pos", "pos", 1],
        [[0, 1], 1.1, "pos", "pos", 0],
    ],
)
def test_threshold_at_tpr(pos, tpr, score_class, equal_class, expected):
    scores = Scores(pos, [], score_class=score_class, equal_class=equal_class)
    threshold = scores.threshold_at_tpr(tpr)
    np.testing.assert_allclose(threshold, expected, atol=1e-10)


@pytest.mark.parametrize(
    "scores, ratio, score_class, equal_class",
    [
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "pos", "pos"],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "pos", "pos"],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "pos", "neg"],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "pos", "neg"],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "neg", "pos"],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "neg", "pos"],
        [[1, 2, 3, 4], [0, 0.25, 0.5, 0.75, 1], "neg", "neg"],
        [[1, 2, 3, 4], [0.125, 0.375, 0.625, 0.875], "neg", "neg"],
        # Simple interpolation
        [[0, 1], 0.6, "pos", "pos"],
        [[0, 1], 0.6, "pos", "neg"],
        [[0, 1], 0.6, "neg", "pos"],
        [[0, 1], 0.6, "neg", "neg"],
        # Edge cases
        [[0, 1], -0.1, "pos", "pos"],
        [[0, 1], 1.1, "pos", "pos"],
    ],
)
def test_threshold_setting(scores, ratio, score_class, equal_class):
    # We reduce the other cases (FNR, TNR and FPR) to TPR calculations
    score_obj = Scores(scores, [], score_class=score_class, equal_class=equal_class)
    expected = score_obj.threshold_at_tpr(ratio)

    # FNR
    score_obj = Scores(scores, [], score_class=score_class, equal_class=equal_class)
    threshold = score_obj.threshold_at_fnr(1. - np.asarray(ratio))
    np.testing.assert_allclose(threshold, expected, atol=1e-10)

    # TNR
    reverse_class = "neg" if equal_class == "pos" else "pos"
    score_obj = Scores([], scores, score_class=score_class, equal_class=reverse_class)
    threshold = score_obj.threshold_at_tnr(1. - np.asarray(ratio))
    np.testing.assert_allclose(threshold, expected, atol=1e-10)

    # FPR
    reverse_class = "neg" if equal_class == "pos" else "pos"
    score_obj = Scores([], scores, score_class=score_class, equal_class=reverse_class)
    threshold = score_obj.threshold_at_fpr(ratio)
    np.testing.assert_allclose(threshold, expected, atol=1e-10)
