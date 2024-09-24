import numpy as np
import pytest

from score_analysis.experimental import (
    BernoulliDataset,
    CorrelatedBernoullilDataset,
    NormalDataset,
)


def test_normal_dataset():
    rng = np.random.default_rng(seed=42)
    generator = NormalDataset(mu_pos=1.0)
    scores = generator.sample(n=200, p_pos=0.4, rng=rng)
    assert abs(scores.nb_all_pos - 80) < 20
    assert abs(scores.nb_all_neg - 120) < 20

    generator.roc(fnr=np.linspace(0, 1, 10))
    generator.roc(fpr=np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        generator.roc(fnr=None, fpr=None)
    with pytest.raises(ValueError):
        generator.roc(fnr=np.array([0, 1]), fpr=np.array([0, 1]))

    # Test both scalar and array versions
    for th in [0.3, np.array([0.2, 0.4])]:
        np.testing.assert_allclose(generator.threshold_at_fnr(generator.fnr(th)), th)
        np.testing.assert_allclose(generator.threshold_at_fpr(generator.fpr(th)), th)


def test_normal_dataset_from_metrics():
    generator = NormalDataset.from_metrics(
        fnr=0.3, fpr=0.4, fnr_support=100, fpr_support=40
    )
    np.testing.assert_allclose(generator.fnr(0.0), 0.3)
    np.testing.assert_allclose(generator.fpr(0.0), 0.4)


def test_bernoulli_dataset():
    rng = np.random.default_rng(seed=42)
    generator = BernoulliDataset(p=0.6)

    data = generator.sample(n=10, random=False)
    assert data.sum() == 6

    data = generator.sample(n=1000, random=True, rng=rng)
    assert abs(data.sum() - 600) <= 10

    with pytest.raises(ValueError):
        generator.sample()  # Not passing any n


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_correlated_bernoulli_dataset(rho):
    rng = np.random.default_rng(seed=42)
    generator = CorrelatedBernoullilDataset(p1=0.9, p2=0.8, rho=rho)

    data = generator.sample(n=100, random=False, rng=rng)
    assert abs(data[0].sum() - 90) <= 2  # Need random False for (almost) exact numbers
    assert abs(data[1].sum() - 80) <= 2
    rho_exact = np.corrcoef(data)[0, 1]
    assert abs(rho_exact - rho) < 0.1

    data = generator.sample(n=1000, random=True, rng=rng)
    assert abs(data[0].sum() - 900) < 20
    assert abs(data[1].sum() - 800) < 20
    rho_exact = np.corrcoef(data)[0, 1]
    assert abs(rho_exact - rho) < 0.1

    with pytest.raises(ValueError):
        generator.sample()  # Not passing any n

    # Invalid parameters, cannot construct probabilities.
    generator = CorrelatedBernoullilDataset(p1=0.9, p2=0.8, rho=0.99)
    with pytest.raises(ValueError, match="negative probabilities"):
        generator.sample(n=100)
