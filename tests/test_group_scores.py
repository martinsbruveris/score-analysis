import numpy as np
import pytest

from score_analysis import BootstrapConfig, GroupScores, Scores, groupwise


@pytest.mark.parametrize("metric", ["tpr", Scores.tpr])
def test_groupwise(metric):
    """Tests the groupwise decorator against some hand-implemented method."""
    scores = GroupScores.from_labels(
        labels=[0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        scores=[0, 5, 1, 2, 6, 0, 1, 4, 1, 6],
        groups=["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
    )
    res_1 = scores.group_tpr(threshold=3.0)
    res_2 = groupwise(metric)(scores, threshold=3.0)
    np.testing.assert_array_equal(res_1, res_2)


def test_from_labels():
    scores = GroupScores.from_labels(
        labels=[1, 1, 0], scores=[3, 2, 1], groups=["a", "b", "b"], pos_label=1
    )
    np.testing.assert_equal(scores.pos, [2, 3])
    np.testing.assert_equal(scores.neg, [1])
    np.testing.assert_equal(scores.pos_groups, ["b", "a"])
    np.testing.assert_equal(scores.neg_groups, ["b"])
    np.testing.assert_equal(scores.groups, ["a", "b"])


def test_swap():
    scores = GroupScores(
        pos=[1, 2],
        neg=[3, 4],
        pos_groups=[8, 9],
        neg_groups=[4, 4],
        score_class="pos",
        equal_class="neg",
    )
    expected = GroupScores(
        pos=[3, 4],
        neg=[1, 2],
        pos_groups=[4, 4],
        neg_groups=[8, 9],
        score_class="neg",
        equal_class="pos",
    )
    assert scores.swap() == expected


def test_getitem():
    scores = GroupScores(
        pos=[1, 2, 3], neg=[5, 6], pos_groups=[0, 1, 0], neg_groups=[1, 0]
    )
    scores_0 = Scores(pos=[1, 3], neg=[6])
    scores_1 = Scores(pos=[2], neg=[5])
    assert scores[0] == scores_0
    assert scores[1] == scores_1


def test_group_cm():
    scores = GroupScores(
        pos=[1, 3, 5], neg=[2, 3], pos_groups=[0, 1, 0], neg_groups=[1, 0]
    )
    cm = scores.group_cm(3.4)

    assert cm[0] == scores[0].cm(3.4)
    assert cm[1] == scores[1].cm(3.4)


def test_bootstrap_sample_non_stratified():
    """Tests non-stratified bootstrap sampling."""
    scores = GroupScores(
        pos=[0, 1, 2], neg=[3, 4], pos_groups=["a", "b", "c"], neg_groups=["d", "e"]
    )
    score_to_group = {d: c for d, c in zip([0, 1, 2, 3, 4], "abcde")}
    sample = scores.bootstrap_sample(
        config=BootstrapConfig(sampling_method="replacement", stratified_sampling=None)
    )

    assert sample.nb_all_samples == scores.nb_all_samples
    # We test that the group correspondence has been preserved
    for s, g in zip(sample.pos, sample.pos_groups):
        assert g == score_to_group[s]
    for s, g in zip(sample.neg, sample.neg_groups):
        assert g == score_to_group[s]


def test_bootstrap_sample_by_group():
    """Tests group-wise stratified bootstrap sampling."""
    scores = GroupScores(
        pos=[0, 1, 2], neg=[3, 4], pos_groups=["a", "b", "c"], neg_groups=["d", "e"]
    )
    sample = scores.bootstrap_sample(
        config=BootstrapConfig(
            sampling_method="replacement", stratified_sampling="by_group"
        )
    )
    # Because there is only one sample per group, the bootstrap sample has to equal the
    # original scores object.
    assert sample == scores


def test_bootstrap_sample_callable():
    """Tests passing custom sampling method."""
    scores = GroupScores(pos=[0, 1, 2], neg=[1], pos_groups=[0, 0, 0], neg_groups=[0])
    sample = scores.bootstrap_sample(
        config=BootstrapConfig(sampling_method=lambda x: x)
    )  # Identity sampling
    assert sample == scores


def test_bootstrap_sample_unsupported():
    """Tests various unsupported bootstrap configurations."""
    scores = GroupScores(pos=[0], neg=[1], pos_groups=[0], neg_groups=[1])

    with pytest.raises(ValueError):
        scores.bootstrap_sample(
            config=BootstrapConfig(
                sampling_method="replacement", stratified_sampling="no_such_sampling"
            )
        )
    with pytest.raises(ValueError):
        scores.bootstrap_sample(
            config=BootstrapConfig(sampling_method="replacement", smoothing=True)
        )
    with pytest.raises(ValueError):
        scores.bootstrap_sample(config=BootstrapConfig(sampling_method="proportion"))
    with pytest.raises(ValueError):
        scores.bootstrap_sample(
            config=BootstrapConfig(sampling_method="no_such_method")
        )
    with pytest.raises(ValueError):
        scores.bootstrap_sample(config=BootstrapConfig(sampling_method=None))


@pytest.mark.parametrize("method", ["quantile", "bc", "bca"])  # Also add bc and bca
def test_bootstrap_ci(method):
    """
    We test that we can compute CIs. While the functionality is inherited from Scores
    (and tested there), we want to make sure, that everything works when put together.
    """
    scores = GroupScores(
        pos=[0, 1, 2], neg=[1, 3], pos_groups=[0, 0, 1], neg_groups=[0, 1]
    )
    ci = scores.bootstrap_ci(
        "group_fpr", threshold=1.2, config=BootstrapConfig(bootstrap_method=method)
    )
    assert np.all(np.isfinite(ci))
