from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.stats

from score_analysis import BinaryLabel, ROCCurve, Scores


@dataclass
class NormalDataset:
    mu_pos: float
    mu_neg: Optional[float] = None
    sigma_pos: float = 3.75
    sigma_neg: float = 3.0
    p_pos: float = 0.5
    n: Optional[int] = None
    score_class: Union[BinaryLabel, str] = "pos"

    def __post_init__(self):
        if self.mu_neg is None:
            self.mu_neg = -self.mu_pos

    @staticmethod
    def from_metrics(
        fnr: float,
        fpr: float,
        fnr_support: int,
        fpr_support: int,
        sigma_pos: float = 1.0,
        sigma_neg: float = 1.0,
    ) -> NormalDataset:
        """
        Generates a dataset with normally distributed scores, such that at
        the threshold 0.0 we obtain the given FNR at FPR and there are
        fnr_support and fpr_support many false negatives and false positives
        respectively.
        """

        # We want P(X < 0) = fnr, where X ~ N(mu, si). We rewrite this as
        #     P((X - mu) / si < -mu / si) = fnr
        # and (X - mu) / si ~ N(0, 1), so -mu / si can be obtained as quantiles
        # of N(0, 1).
        mu_pos = -scipy.stats.norm.ppf(fnr) * sigma_pos
        nb_pos = int(fnr_support / fnr)

        # Similarly, we rewrite P(X < 0) = 1 - fpr as
        #     P((X - mu) / si < -mu / si) = 1 - fpr
        mu_neg = -scipy.stats.norm.ppf(1 - fpr) * sigma_neg
        nb_neg = int(fpr_support / fpr)

        n = nb_pos + nb_neg
        p_pos = nb_pos / n

        return NormalDataset(
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sigma_pos=sigma_pos,
            sigma_neg=sigma_neg,
            p_pos=p_pos,
            n=n,
            score_class="pos",
        )

    def sample(
        self,
        n: Optional[int] = None,
        *,
        p_pos: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Scores:
        n = n if n is not None else self.n
        p_pos = p_pos if p_pos is not None else self.p_pos
        rng = rng or np.random.default_rng()
        nb_pos = rng.binomial(n, p_pos)
        nb_neg = n - nb_pos
        pos = rng.normal(loc=self.mu_pos, scale=self.sigma_pos, size=nb_pos)
        neg = rng.normal(loc=self.mu_neg, scale=self.sigma_neg, size=nb_neg)
        return Scores(pos=pos, neg=neg, score_class=self.score_class)

    def roc(
        self,
        *,
        fnr: Optional[np.ndarray] = None,
        fpr: Optional[np.ndarray] = None,
    ) -> ROCCurve:
        if fnr is None and fpr is None:
            raise ValueError("Must provide either FNR or FPR.")
        if fnr is not None and fpr is not None:
            raise ValueError("Cannot provide both FNR and FPR.")

        if fnr is not None:
            threshold = scipy.stats.norm.ppf(fnr, loc=self.mu_pos, scale=self.sigma_pos)
        else:
            # isf is the inverse survival function (sf = 1 - cdf)
            threshold = scipy.stats.norm.isf(fpr, loc=self.mu_neg, scale=self.sigma_neg)

        fnr = scipy.stats.norm.cdf(threshold, loc=self.mu_pos, scale=self.sigma_pos)
        fpr = scipy.stats.norm.sf(threshold, loc=self.mu_neg, scale=self.sigma_neg)

        return ROCCurve(fnr=fnr, fpr=fpr)

    def threshold_at_fnr(self, fnr: np.ndarray) -> np.ndarray:
        threshold = scipy.stats.norm.ppf(fnr, loc=self.mu_pos, scale=self.sigma_pos)
        if np.isscalar(fnr):
            threshold = threshold.item()
        return threshold

    def threshold_at_fpr(self, fpr: np.ndarray) -> np.ndarray:
        # isf is the inverse survival function (sf = 1 - cdf)
        threshold = scipy.stats.norm.isf(fpr, loc=self.mu_neg, scale=self.sigma_neg)
        if np.isscalar(fpr):
            threshold = threshold.item()
        return threshold

    def fnr(self, threshold: np.ndarray) -> np.ndarray:
        fnr = scipy.stats.norm.cdf(threshold, loc=self.mu_pos, scale=self.sigma_pos)
        if np.isscalar(threshold):
            fnr = fnr.item()
        return fnr

    def fpr(self, threshold: np.ndarray) -> np.ndarray:
        fpr = scipy.stats.norm.sf(threshold, loc=self.mu_neg, scale=self.sigma_neg)
        if np.isscalar(threshold):
            fpr = fpr.item()
        return fpr


@dataclass
class CorrelatedBernoullilDataset:
    """
    Dataset of two bernoulli random variables with a fixed correlation coefficient.

    Args:
        p1: Success probability for first random variable.
        p2: Success probability for second random variable.
        rho: Correlation coefficient.
        n: Optional dataset size.
    """

    p1: float
    p2: float
    rho: float
    n: Optional[int] = None

    def sample(
        self,
        n: Optional[int] = None,
        *,
        random: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample a dataset.

        Args:
            n: Dataset size. If not provided, we use the one specified in the class.
            random: If False, we aim to have proportion of successes in the dataset to
                be as close to ``p1`` and ``p2`` as possible. If True, we sample from
                a binomial distribution with these parameters.
            rng: Random number generator.

        Returns:
            Array of shape (2, n) with elements 0 or 1.
        """
        # For the algorithm see:
        #     https://stats.stackexchange.com/questions/284996/
        #     generating-correlated-binomial-random-variables
        #
        # We sample from a joint distribution with the following parameters
        #   P((X,Y) = (0, 0)) = a
        #   P((X,Y) = (1, 0)) = 1 - q - a
        #   P((X,Y) = (0, 1)) = 1 - p - a
        #   P((X,Y) = (1, 1)) = a + p + q - 1 ,
        # and we compute `a` via the formula
        #   a = (1-p)*(1-q) + rho * sqrt(p*q*(1-p)*(1-q))
        p1 = self.p1
        p2 = self.p2

        n = n or self.n
        if n is None:
            raise ValueError("Dataset size n cannot be None.")
        rng = rng or np.random.default_rng()

        c = (1 - p1) * (1 - p2)
        a = c + self.rho * np.sqrt(p1 * p2 * c)
        # 0 .. (0, 0), 1 .. (1, 0), 2 .. (0, 1), 3 .. (1, 1)
        p = np.array([a, 1 - p2 - a, 1 - p1 - a, p1 + p2 + a - 1])

        if np.any(p < 0):
            raise ValueError("Dataset parameters lead to negative probabilities.")

        if random:
            joint = rng.choice(4, size=n, p=p)
        else:
            nb = np.floor(n * p).astype(int)
            nb[-1] = n - np.sum(nb[:-1])
            joint = np.repeat(np.arange(4), nb)
            # The number of 0s and 1s is fixed, but the order is still random.
            rng.shuffle(joint)

        # Now combine into (2, n) array
        data = np.empty((2, n), dtype=int)
        data[0] = joint % 2
        data[1] = joint // 2

        return data
