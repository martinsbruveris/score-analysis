"""
This module contains internal utility functions.
"""
from typing import Optional, Union

import numpy as np
import scipy.stats


def binomial_ci(count: np.ndarray, nobs: np.ndarray, alpha: float = 0.05):
    """
    Confidence interval for binomial proportions.

    Args:
        count: Number of successes of shape (...)
        nobs: Number of trials of shape (...)
        alpha: Significance level. In range (0, 1)

    Returns:
        np.ndarray of shape (..., 2). Lower and upper limits of confidence interval
        with coverage 1-alpha.
    """
    nans = np.full_like(count, np.nan, dtype=float)
    p = np.divide(count, nobs, out=nans, where=nobs != 0)

    nans = np.full_like(count, np.nan, dtype=float)
    std = np.divide(p * (1 - p), nobs, out=nans, where=nobs != 0)
    std = np.sqrt(std)

    dist = scipy.stats.norm.isf(alpha / 2.0) * std
    ci = np.stack([p - dist, p + dist], axis=-1)
    return ci


def bootstrap_ci(
    theta: np.ndarray,
    theta_hat: Optional[Union[float, np.ndarray]] = None,
    alpha: float = 0.05,
    *,
    method: str = "quantile",
) -> np.ndarray:
    """
    Calculates the bootstrap confidence interval with approximate coverage 1-alpha for
    the empirical sample theta. We assume that we have computed N bootstrap estimates,
    theta, of the quantity of interest, theta_hat.

    This function then constructs a confidence interval based on the bootstrap
    estimates, using the bias corrected and accelerated quantile method.

    Args:
        theta: Array of shape (N, Y), where N is the number of samples.
        theta_hat: Array of shape (Y,) with the empirical estimate of the metric. This
            is only needed for the methods "bc" and "bca".
        alpha: Significance level. In range (0, 1).
        method: Method to compute the CI from the bootstrap samples.
            Possible values are

            * "quantile" uses the alpha/2 and 1-alpha/2 quantiles of the
              empirical metric distribution.
            * "bc" applies bias correction to correct for the bias of the median
              of the empirical distribution
            * "bca" applies bias correction and acceleration to correct for non-
              constant standard error.

            See Ch. 11 of Computer Age Statistical Inference by Efron and Hastie
            for details.

    Returns:
        Returns an array of shape (Y, 2) with lower and upper bounds of the CI.
    """
    alpha_lower = alpha / 2.0
    alpha_upper = 1 - alpha / 2.0

    if method == "quantile":
        ci = np.quantile(theta, q=[alpha_lower, alpha_upper], axis=0)  # (2, Y)
        ci = np.moveaxis(ci, source=0, destination=-1)  # (Y, 2)
    elif method in {"bc", "bca"}:
        if theta_hat is None:
            raise ValueError(f"Must provide theta_hat when using method {method}.")
        theta_hat = theta_hat[np.newaxis]  # (1, Y)

        # Flatten the metric shape to a vector
        nb_samples = theta.shape[0]
        metric_shape = theta.shape[1:]
        theta = np.reshape(theta, (nb_samples, -1))
        theta_hat = np.reshape(theta_hat, (1, -1))
        metric_size = theta.shape[-1]

        # Proportion of samples less than theta_hat
        p0 = np.sum(theta <= theta_hat, axis=0) / nb_samples  # (Y,)
        # Inverse function of standard normal cdf. If p0=0.5, then z0=0 and this
        # method reduces to the quantile method.
        z0 = scipy.stats.norm.ppf(p0)

        z_alpha_lower = scipy.stats.norm.ppf(alpha_lower)
        z_alpha_upper = scipy.stats.norm.ppf(alpha_upper)

        if method == "bc":
            # See (11.33) in Efron, Hastie
            z_lower = 2 * z0 + z_alpha_lower
            z_upper = 2 * z0 + z_alpha_upper
        else:  # bootstrap_method == "bca"
            # Estimate acceleration from data. See (11.40) in Efron, Hastie.
            a_num = np.sum((theta - theta_hat) ** 3, axis=0)
            a_den = 6 * np.sum((theta - theta_hat) ** 2, axis=0) ** 1.5
            # If a=0, the method reduces to the non-accelerated bias correction
            a = np.divide(a_num, a_den, out=np.zeros_like(a_num), where=a_den != 0)
            # See (11.39) in Efron, Hastie
            fin = np.isfinite(z0)
            z_lower = np.copy(z0)
            s_lower = z0[fin] + z_alpha_lower
            z_lower[fin] = z0[fin] + s_lower / (1 - a[fin] * s_lower)
            z_upper = np.copy(z0)
            s_upper = z0[fin] + z_alpha_upper
            z_upper[fin] = z0[fin] + s_upper / (1 - a[fin] * s_upper)

        alpha_hat_lower = scipy.stats.norm.cdf(z_lower)
        alpha_hat_upper = scipy.stats.norm.cdf(z_upper)

        ci = np.empty((metric_size, 2))
        for j in range(metric_size):
            ci[j] = np.quantile(
                theta[:, j], q=[alpha_hat_lower[j], alpha_hat_upper[j]], axis=0
            )
        ci = np.reshape(ci, (*metric_shape, 2))
    else:
        raise ValueError(f"Unknown value for bootstrap_method: {method}")

    return ci
