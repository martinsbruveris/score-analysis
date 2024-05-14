Confidence Intervals
====================

The library uses bootstrapping to compute confidence intervals for arbitrary metrics,
without relying on prior assumptions on their statistical distribution.

For example, we can compute the confidence interval around EER and the EER threshold.

.. code-block:: python

    import numpy as np
    from score_analysis import Scores

    scores = Scores(
        pos=np.random.normal(loc=1.0, scale=2.0, size=10_000),
        neg=np.random.normal(loc=-1.0, scale=1.5, size=10_000),
    )

    theta_ci, eer_ci = scores.bootstrap_ci(metric=Scores.eer, alpha=0.05)

    print(f"Threshold 95%-CI: ({theta_ci[0]:.4f}, {theta_ci[1]:.4f})")
    print(f"EER 95%-CI: ({eer_ci[0]:.3%}, {eer_ci[1]:.3%})")

This results in the output

.. code-block::

    Threshold 95%-CI: (-0.1775, -0.1214)
    EER 95%-CI: (27.287%, 28.512%)


Bootstrapping
-------------

Given a set of scores :math:`S`, bootstrapping calculates the confidence interval of an
arbitrary metric :math:`f` using the following algorithm:

* From :math:`S`, sample a new set of scores :math:`S_i` of the same size as :math:`S`,
  using sampling with replacement.
* Compute the value of the metric :math:`y_i = f(S_i)` on the sampled dataset.
* Repeat the process :math:`n` times to obtain :math:`n` values :math:`{y_i}` for the
  metric of interest. This is the set of bootstrapped measurements.
* The confidence interval at significance level :math:`\alpha` is given by the
  :math:`\alpha/2`-quantiles and :math:`1-\alpha/2`-quantiles of the set :math:`{y_i}`.

There are several parameters we can vary in the process and this library allows
the user to vary most of them.

* The number :math:`n` of bootstrapped measurements to compute. A sensible
  number is 1,000. It strikes a balance between stable results and a tractable
  computational time.
* The sampling method used to create the bootstrap sample :math:`S_i`. Instead of
  replacement sampling across all scores, we can use stratified sampling, or a faster
  approximation to replacement sampling.
* The quantile method to compute the confidence interval given a set of bootstrapped
  measurements :math:`{y_i}` can be refined by adding bias correction and acceleration.

We describe several of the features provided by the ``score_analysis`` library below.

Sampling
--------

Replacement sampling
^^^^^^^^^^^^^^^^^^^^

The default sampling method is sampling with replacement across all scores. We first
determine, using a binomial random variable, the number of hard positive and negative
scores in the bootstrap sample and then sample the scores themselves. This means that
the number of positive and negative scores (both easy and hard) will vary across
bootstrap samples. To avoid problems with computing metrics we ensure that each
bootstrap sample has at least a positive and a negative score if the source had them
as well.

Single-pass sampling
^^^^^^^^^^^^^^^^^^^^

Replacement sampling uses ``np.random.choice`` to select the scores for the sample.
Since the function returns an unsorted array, each bootstrap sample requires sorting
the selected positive and negative scores, even though we start with already sorted
scores.

To avoid the need to sort each sample, we can observe that the number of times each
individual score is included in the bootstrap sample follows a binomial distribution.
So, to construct a bootstrap sample, we first determine, how often each score will
be included and then select those scores using `np.repeat`. This speeds up bootstrapping
by about 40%.

This code sample explain the core mechanic. Given a sorted array ``scores``, we
construct a sorted array ``sample``.

.. code-block:: python

    n = len(scores)
    nb_included = np.random.binomial(size=n, n=n, p=1. / n)
    sample = np.repeat(scores, nb_included)

Because we choose for each sample independently, how often to include it in the
bootstrap sample, there is no guarantee that the overall sample will have the same
size as the original scores. But this should not be a problem in practice when
we have 100+ scores available.

Dynamic sampling strategy
^^^^^^^^^^^^^^^^^^^^^^^^^

The dynamic sampling strategy selects between

* replacement sampling and
* single-pass sampling

depending on the number scores. It selects single-pass sampling, if there are at least
100 positive and negative scores available and falls back to replacement sampling
otherwise. Thus the user can benefit from faster bootstrapping when it is safe to do
so.

Custom sampling method
^^^^^^^^^^^^^^^^^^^^^^

The user can also provide their own custom sampling method. A sampling method is a
function

.. code-block:: python

    def sampling_method(scores: Scores) -> Scores:
        ...

that takes a ``Scores`` object as input and returns a ``Scores`` object of the sample.
