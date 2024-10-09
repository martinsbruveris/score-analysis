"""
This module contains various utility functions used elsewhere in the library.
"""

from typing import List, Optional, Union

import numpy as np
import scipy.stats


def binomial_ci(count: np.ndarray, nobs: np.ndarray, alpha: float = 0.05):
    """
    Confidence interval for binomial proportions using the normal approximation.

    Args:
        count: Number of successes of shape (X,).
        nobs: Number of trials of shape (X,).
        alpha: Significance level. In range (0, 1)

    Returns:
        np.ndarray of shape (X, 2). Lower and upper limits of confidence interval with
        coverage 1-alpha.
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
    alpha: Optional[Union[float, np.ndarray]] = 0.05,
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
        alpha: Significance level. In range (0, 1). Vector-valued alpha only supported
            with "quantile" method.
        method: Method to compute the CI from the bootstrap samples.
            Possible values are

            * "quantile" uses the alpha/2 and 1-alpha/2 quantiles of the
              empirical metric distribution.
            * "bc" applies bias correction to correct for the bias of the median
              of the empirical distribution
            * "bca" applies bias correction and acceleration to correct for
              non-constant standard error.

            See Ch. 11 of Computer Age Statistical Inference by Efron and Hastie
            for details.

    Returns:
        Returns an array of shape (Y, 2) with lower and upper bounds of the CI. For
        vector-valued alpha of shape (Z,) the return values has shape (Y, Z, 2).
    """
    alpha = np.asarray(alpha)  # (Z,)
    alpha_shape = alpha.shape
    alpha = np.reshape(alpha, -1)  # (Z',)

    alpha_lower = alpha / 2.0
    alpha_upper = 1 - alpha / 2.0

    if method == "quantile":
        alpha_joint = np.stack([alpha_lower, alpha_upper], axis=0)  # (2, Z')
        ci = np.nanquantile(theta, q=alpha_joint, axis=0)  # (2, Z', Y)
        ci = np.moveaxis(ci, source=[0, 1], destination=[-1, -2])  # (Y, Z', 2)
        ci = np.reshape(ci, theta.shape[1:] + alpha_shape + (2,))  # (Y, Z, 2)
    elif method in {"bc", "bca"}:
        if theta_hat is None:
            raise ValueError(f"Must provide theta_hat when using method {method}.")
        theta_hat = np.asarray(theta_hat)
        theta_hat = theta_hat[np.newaxis]  # (1, Y)

        # Flatten the metric shape to a vector
        nb_samples = theta.shape[0]
        metric_shape = theta.shape[1:]
        theta = np.reshape(theta, (nb_samples, -1))
        theta_hat = np.reshape(theta_hat, (1, -1))
        metric_size = theta.shape[-1]

        # Proportion of samples less than theta_hat
        nb_not_nan = np.sum(~np.isnan(theta), axis=0)
        p0 = np.sum(theta <= theta_hat, axis=0) / nb_not_nan  # (Y,)
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
            a_num = np.nansum((theta - theta_hat) ** 3, axis=0)
            a_den = 6 * np.nansum((theta - theta_hat) ** 2, axis=0) ** 1.5
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
            ci[j] = np.nanquantile(
                theta[:, j], q=[alpha_hat_lower[j], alpha_hat_upper[j]], axis=0
            )
        ci = np.reshape(ci, (*metric_shape, 2))
    else:
        raise ValueError(f"Unknown value for bootstrap_method: {method}")

    return ci


def invert_pl_function(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
    """
    Inverts piecewise linear function.

    The piecewise linear function  :math:`f(x_i) = y_i` is defined the by the sequence
    of points :math:`(x_i, y_i)`. We assume that :math:`x` is an increasing
    vector, i.e., :math:`x_i \leq x_{i+1}`; the vector :math:`x` does not have to be
    strictly increasing, but if :math:`x_i = x_{i+1}`, then we assume that also
    :math:`y_i = y_{i+1}`.

    For a given target value :math:`t_j`, the function finds all values
    :math:`s_{j, k}`, such that :math:`f(s_{j, k}) = t_j`. Because the number
    of solutions can vary for different :math:`t_j`, the return type for :math:`s`
    is a list of the same length as :math:`t` with each entry an array.

    If the equation :math:`f(s) = t_j` has no solution, we return the closest point
    :math:`s_j`, i.e.,

    .. math::

        \|f(s_j) - t_j\| = \min_z \|f(z) - t_j\|\,.
    We always return only one solution in this case.

    Args:
         x: Increasing vector of points defining the function.
         y: Vector of same length as x defining the function values.
         t: Vector of points at which to invert the function.

    Returns:
        A list :math:`s` of the same length as :math:`t` of arrays such that for all
        :math:`j`, the array :math:`s_j` is strictly increasing and contains all
        solutions of the equation :math:`f(s) = t_j`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)
    t_scalar = t.ndim == 0  # Is input a scalar?

    x = x[:, np.newaxis]  # (N, 1)
    y = y[:, np.newaxis]  # (N, 1)
    # We turn scalar t into a 1-dim array + extra dimension for broadcasting
    t = t[np.newaxis, np.newaxis] if t_scalar else t[np.newaxis, :]  # (1, T)

    # Find out where we cross the threshold
    crossing_up = (y[:-1] <= t) & (y[1:] > t)
    crossing_down = (y[:-1] >= t) & (y[1:] < t)
    crossing = crossing_up | crossing_down
    # Get indices of crossings; Note that np.nonzero tests the array in row-major
    # order, meaning that after transposing for each t index, the s indices will
    # be already in increasing order.
    t_indices, s_indices = np.nonzero(crossing.T)

    # Find closest points to given t values in case we don't find exact solutions.
    min_ind = np.argmin(np.abs(y - t), axis=0)  # (T,)
    s_min = x[min_ind]

    t = t[0]  # We are done with broadcasting
    x = x[:, 0]
    y = y[:, 0]

    s = [[] for _ in t]
    # Find all solutions given the crossing indices
    for t_ind, j in zip(t_indices, s_indices):
        # Apply linear interpolation between x[j] and x[j+1]
        la = (t[t_ind] - y[j]) / (y[j + 1] - y[j])
        z = (1 - la) * x[j] + la * x[j + 1]
        s[t_ind].append(z)

    # Deal with cases where we don't have solutions
    for j in range(len(s)):
        if len(s[j]) == 0:
            s[j].append(s_min[j])

    # Convert to list of arrays
    s = [np.asarray(z) for z in s]

    if t_scalar:  # Reduce back to scalar
        s = s[0]

    return s
