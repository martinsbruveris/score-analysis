import numpy as np
import pytest

from score_analysis.scores import Scores, pointwise_cm


def test_from_labels():
    scores = Scores.from_labels(labels=[2, 2, 3], scores=[3, 2, 1], pos_label=2)
    np.testing.assert_equal(scores.pos, [2, 3])
    np.testing.assert_equal(scores.neg, [1])


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
def test_cm(pos, neg, threshold, score_class, equal_class, expected):
    scores = Scores(pos, neg, score_class=score_class, equal_class=equal_class)
    cm = scores.cm(threshold)
    np.testing.assert_equal(cm.matrix, expected)


@pytest.mark.parametrize(
    "pos, neg, threshold, expected",
    [
        # Simple scalar version
        [[0, 1, 2, 3], [0, 1, 2], 1.5, 0.5],
        # Vectorized version
        [[0, 1, 2, 3], [], [-1, 0, 1.5, 3, 4], [1.0, 1.0, 0.5, 0.25, 0.0]],
    ],
)
def test_tpr(pos, neg, threshold, expected):
    scores = Scores(pos, neg)
    tpr = scores.tpr(threshold)
    np.testing.assert_equal(tpr, expected)


def test_fnr_etc():
    scores = Scores(pos=[0, 1, 2, 3], neg=[0, 1, 2, 3, 4])
    tpr = scores.tpr(threshold=2.5)
    np.testing.assert_allclose(tpr, 0.25)

    fnr = scores.fnr(threshold=2.5)
    np.testing.assert_allclose(fnr, 0.75)

    tnr = scores.tnr(threshold=2.5)
    np.testing.assert_allclose(tnr, 0.6)

    fpr = scores.fpr(threshold=2.5)
    np.testing.assert_allclose(fpr, 0.4)


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


def test_invalid_threshold_at_tpr_etc():
    with pytest.raises(ValueError):
        Scores(pos=[], neg=[1, 2]).threshold_at_tpr(0.5)
    with pytest.raises(ValueError):
        Scores(pos=[], neg=[1, 2]).threshold_at_fnr(0.5)
    with pytest.raises(ValueError):
        Scores(pos=[1, 2], neg=[]).threshold_at_tnr(0.5)
    with pytest.raises(ValueError):
        Scores(pos=[1, 2], neg=[]).threshold_at_fpr(0.5)


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
    threshold = score_obj.threshold_at_fnr(1.0 - np.asarray(ratio))
    np.testing.assert_allclose(threshold, expected, atol=1e-10)

    # TNR
    reverse_class = "neg" if equal_class == "pos" else "pos"
    score_obj = Scores([], scores, score_class=score_class, equal_class=reverse_class)
    threshold = score_obj.threshold_at_tnr(1.0 - np.asarray(ratio))
    np.testing.assert_allclose(threshold, expected, atol=1e-10)

    # FPR
    reverse_class = "neg" if equal_class == "pos" else "pos"
    score_obj = Scores([], scores, score_class=score_class, equal_class=reverse_class)
    threshold = score_obj.threshold_at_fpr(ratio)
    np.testing.assert_allclose(threshold, expected, atol=1e-10)


@pytest.mark.parametrize(
    "pos, neg, score_class, expected_threshold, expected_eer",
    [
        [[0, 1, 1, 1], [0, 1, 1, 1], "pos", 1.0, 0.375],
        # Perfect separation
        [[2, 3, 4], [0, 1], "pos", 1.5, 0.0],
        [[0, 1], [3, 4], "neg", 2.0, 0.0],
        # Perfect separation, but in the wrong direction
        [[2, 3, 4], [5, 6], "pos", 4.5, 1.0],
        [[5, 6], [1, 2], "neg", 3.5, 1.0],
    ],
)
def test_eer(pos, neg, score_class, expected_threshold, expected_eer):
    scores = Scores(pos, neg, score_class=score_class)
    threshold, eer = scores.eer()
    np.testing.assert_allclose(threshold, expected_threshold)
    np.testing.assert_allclose(eer, expected_eer)


def test_find_root_invalid_input():
    # Tests raising exception on invalid input function.
    with pytest.raises(ValueError):
        Scores._find_root(lambda _: 1.0, 0.0, 1.0, True)


#
def test_bootstrap_sample_replacement():
    scores = Scores(
        pos=[1, 2, 3, 4], neg=[1, 2, 3], score_class="neg", equal_class="neg"
    )

    with pytest.raises(ValueError):
        scores.bootstrap_sample(method="no_such_method")

    sample = scores.bootstrap_sample(method="replacement")
    assert sample.pos.size == scores.pos.size
    assert sample.neg.size == scores.neg.size
    assert set(sample.pos).issubset(set(scores.pos))
    assert set(sample.neg).issubset(set(scores.neg))
    assert sample.score_class == scores.score_class
    assert sample.equal_class == scores.equal_class


def test_bootstrap_sample_proportion():
    scores = Scores(pos=[1, 2, 3, 4, 5, 6], neg=[1, 2, 3, 4])

    with pytest.raises(ValueError):
        scores.bootstrap_sample(method="proportion", ratio=None)

    sample = scores.bootstrap_sample(method="proportion", ratio=0.5)
    assert sample.pos.size == 3
    assert sample.neg.size == 2
    assert set(sample.pos).issubset(set(scores.pos))
    assert set(sample.neg).issubset(set(scores.neg))


def test_bootstrap_sample_callable():
    scores = Scores(pos=[1, 2, 3, 4], neg=[1, 2, 3])

    with pytest.raises(ValueError):
        scores.bootstrap_sample(method=None)

    sample = scores.bootstrap_sample(lambda x: x)  # Identity sampling
    assert sample == scores


@pytest.mark.parametrize("metric", ["eer", Scores.eer])
@pytest.mark.parametrize("nb_samples", [1, 3])
def test_bootstrap_metric(metric, nb_samples):
    scores = Scores(pos=[1, 2, 3, 4], neg=[1, 2, 3])
    samples = scores.bootstrap_metric(metric, nb_samples=nb_samples)
    assert samples.shape == (nb_samples, 2)


def test_bootstrap_ci_identity():
    scores = Scores(pos=[1, 2, 3, 4], neg=[1, 2, 3])
    nb_samples = 3
    # Testing with identity sampling, in which case CI should collapse
    ci = scores.bootstrap_ci(
        metric="eer", nb_samples=nb_samples, sampling_method=lambda x: x
    )
    eer = scores.eer()
    for j in range(nb_samples):
        np.testing.assert_equal(ci[..., 0], eer)
        np.testing.assert_equal(ci[..., 1], eer)

    # Test invalid bootstrap method
    with pytest.raises(ValueError):
        scores.bootstrap_ci(metric="eer", bootstrap_method="no_such_method")


@pytest.mark.parametrize("bootstrap_method", ["quantile", "bc", "bca"])
def test_bootstrap_ci_gaussian(bootstrap_method):
    rng = np.random.default_rng(seed=42)
    nb_inside = 0
    for j in range(100):
        scores = Scores(pos=rng.normal(size=100), neg=[])
        ci = scores.bootstrap_ci(
            metric=lambda s: np.mean(s.pos),
            bootstrap_method=bootstrap_method,
            # We use a custom sampling method to make the test deterministic by fixing
            # the random number generator
            sampling_method=lambda s: Scores(
                pos=rng.choice(s.pos, size=s.pos.size, replace=True), neg=[]
            ),
            nb_samples=200,
        )
        nb_inside += ci[0] < 0 < ci[1]
    # If the test starts failing, we should check if it is due to flakiness of the RNG.
    # There is a balance between accuracy and how long the tests take to run.
    assert 92 < nb_inside < 98


def test_pointwise_cm():
    cm = pointwise_cm(
        labels=[1, 1, 0, 0],
        scores=[1, 0, 1, 0],
        threshold=1.0,
        score_class="pos",
        equal_class="pos",
    )
    assert cm.shape == (4, 2, 2)
    np.testing.assert_equal(cm[0], [[1, 0], [0, 0]])
    np.testing.assert_equal(cm[1], [[0, 1], [0, 0]])
    np.testing.assert_equal(cm[2], [[0, 0], [1, 0]])
    np.testing.assert_equal(cm[3], [[0, 0], [0, 1]])

    cm = pointwise_cm(
        labels=[1, 1, 0, 0],
        scores=[1, 0, 1, 0],
        threshold=0.0,
        score_class="pos",
        equal_class="neg",
    )
    np.testing.assert_equal(cm[0], [[1, 0], [0, 0]])
    np.testing.assert_equal(cm[1], [[0, 1], [0, 0]])
    np.testing.assert_equal(cm[2], [[0, 0], [1, 0]])
    np.testing.assert_equal(cm[3], [[0, 0], [0, 1]])

    cm = pointwise_cm(
        labels=[1, 1, 0, 0],
        scores=[1, 0, 1, 0],
        threshold=0.0,
        score_class="neg",
        equal_class="pos",
    )
    np.testing.assert_equal(cm[0], [[0, 1], [0, 0]])
    np.testing.assert_equal(cm[1], [[1, 0], [0, 0]])
    np.testing.assert_equal(cm[2], [[0, 0], [0, 1]])
    np.testing.assert_equal(cm[3], [[0, 0], [1, 0]])

    cm = pointwise_cm(
        labels=[1, 1, 0, 0],
        scores=[1, 0, 1, 0],
        threshold=1.0,
        score_class="neg",
        equal_class="neg",
    )
    np.testing.assert_equal(cm[0], [[0, 1], [0, 0]])
    np.testing.assert_equal(cm[1], [[1, 0], [0, 0]])
    np.testing.assert_equal(cm[2], [[0, 0], [0, 1]])
    np.testing.assert_equal(cm[3], [[0, 0], [1, 0]])


def test_pointwise_cm_shape():
    cm = pointwise_cm(
        labels=np.zeros((3, 4)), scores=np.zeros((3, 4)), threshold=np.zeros((5, 1))
    )
    assert cm.shape == (3, 4, 5, 1, 2, 2)
