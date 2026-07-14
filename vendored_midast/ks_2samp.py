# Functions `mecdf` and `ks_2samp` cloned from:
# https://github.com/o-laurent/multivariate-ks-test (written by Olivier Laurent)
#
# Here we habve modification in ks_2samp that enables to get KS_statistic, KS_pvalue, KS_critical_val
# not only testing hypothesis result (True/False)

from typing import Any, Tuple

import numpy as np


def boundary_for_pval(value):
    return np.max([0, np.min([value, 1])])


def p_value_from_crit_value_one_sample(t_s, n, d):
    return boundary_for_pval(d * (n + 1) * np.exp(-2 * n * (t_s**2)))


def p_value_from_crit_value_two_sample(t_s, n, d):
    return boundary_for_pval(d * (n + 1) * np.exp(-n / 2 * (t_s**2)))


def mecdf(x_val: np.ndarray, t: np.ndarray) -> float:
    """Computes the multivariate empirical cdf of x_val at t.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        t: A numpy array of shape (num_samples_t, dim) representing the point at which to evaluate
            the cdf.

    Returns:
        The multivariate empirical cdf of x_val at t.
    """
    lower = (x_val <= t) * 1.0
    return np.mean(np.prod(lower, axis=1))


def _mecdf_batch(x_val: np.ndarray, t_batch: np.ndarray) -> np.ndarray:
    """Vectorised mecdf evaluated at MANY query points at once. Bit-exact with
    calling ``mecdf`` in a loop over ``t_batch`` (verified to 0.0 abs diff),
    but replaces thousands of per-point Python/numpy calls with one
    per-dimension broadcast -- ~7x faster end to end (dev investigation)."""
    n, dim = x_val.shape
    ok = np.ones((n, t_batch.shape[0]), dtype=bool)
    for d in range(dim):
        ok &= x_val[:, d][:, None] <= t_batch[:, d][None, :]
    return ok.mean(axis=0)


def ks_2samp(
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    asymptotic: bool = False,
    verbose: bool = False,
) -> Tuple[float, Any, Any, Any]:
    """Performs a multivariate two-sample extension of the Kolmogorov-Smirnov test.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the first sample.
        y_val: A numpy array of shape (num_samples_y, dim) representing the second sample.
        alpha: The significance level.
        asymptotic: Whether to use the asymptotic approximation or not.
        verbose: Whether to print the test statistic and the critical value.

    Returns:
        A boolean indicating whether the null hypothesis is rejected.
    """
    num_samples_x, dim = x_val.shape
    num_samples_y, num_feats_y = y_val.shape

    if dim != num_feats_y:
        raise ValueError("The two samples do not have the same number of features.")

    diff = np.zeros((num_samples_x, dim))
    idx_desc = num_samples_x - np.arange(num_samples_x)
    for h in range(dim):
        ind = np.argsort(x_val[:, h])[::-1]
        temp = np.take(x_val, ind, axis=0)
        # z[j, i] = max(temp[j:, i]) (suffix max per dimension i), vectorised
        # replacement for the original per-(i,j) Python double loop.
        z_h = np.maximum.accumulate(temp[::-1], axis=0)[::-1]

        Fx = _mecdf_batch(x_val, z_h)
        Fy = _mecdf_batch(y_val, z_h)
        rank_ok = np.round(num_samples_x * Fx).astype(int) == idx_desc
        diff[:, h] = np.abs(Fx - Fy) * rank_ok
        if h == 0:
            Fx0 = _mecdf_batch(x_val, x_val)
            Fy0 = _mecdf_batch(y_val, x_val)
            diff[:, 0] = np.maximum(diff[:, 0], np.abs(Fx0 - Fy0))
    KS = np.max(diff)

    if asymptotic:
        KS_critical_val = np.sqrt(-np.log(alpha / (4 * dim)) * (0.5 / num_samples_x)) + np.sqrt(
            (-1) * np.log(alpha / (4 * dim)) * (0.5 / num_samples_y)
        )
    else:
        KS_critical_val = np.sqrt(-np.log(alpha / (2 * (num_samples_x + 1) * dim)) * (0.5 / num_samples_x)) + np.sqrt(
            (-1) * np.log(alpha / (2 * (num_samples_y + 1) * dim)) * (0.5 / num_samples_y)
        )

    if verbose:
        print("test statistic: ", KS)
        print("test statistic critical value: ", KS_critical_val)

    return float(KS), KS_critical_val, KS_critical_val < KS, p_value_from_crit_value_two_sample(t_s=KS, n=num_samples_x, d=dim)
