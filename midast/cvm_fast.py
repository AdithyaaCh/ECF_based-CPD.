"""Vectorized, R-free re-implementation of MIDAST's CramerTest.

Author: Adithya Cheruvu

MIDAST's CramerTest (multivariate_tests_from_R.py) calls R's
cramer.test(kernel="phiLog") through rpy2 for every sliding-window
comparison. That call recomputes the full pairwise distance matrix from
scratch for every bootstrap replicate and every window position, and pays a
Python/R marshalling cost on top. Since a bootstrap replicate only relabels
which pooled points belong to which group -- the pairwise distances between
points never change -- the distance matrix can be computed once per window
and reused for every replicate, and all replicates can be evaluated together
as a single matrix product against a batched group-indicator matrix.

This gives the same statistic as R's cramer.test(kernel="phiLog") (verified
against R to 5+ decimal places, both for the observed split and for permuted
splits) while removing the R dependency and running roughly two orders of
magnitude faster.

Two entry points:
  - FastCramerTest: drop-in replacement for CramerTest in
    multivariate_tests_from_R.py, same constructor/`conduct_test` signature.
  - fast_cramer_scan: drop-in replacement for
    ChangeDetector(test_name="CramerTest").fit(...), returning a results_df
    with the same columns analyze_results() expects.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def phi_log(d):
    """R's cramer.test(kernel="phiLog") applies phi to the SQUARED Euclidean
    distance: phiLog(t) = log(1 + t), t = ||x - y||^2."""
    return np.log1p(d ** 2)


def energy_stat_batch(Dphi, group_a_mat):
    """Baringhaus-Franz energy statistic for many group splits at once.

    Dphi: (n, n) phi-transformed pairwise distance matrix for the pooled
        sample. group_a_mat: (n, B) boolean matrix, column b is the
        indicator of which of the n pooled points belong to "group A" for
        split b. Returns the statistic for all B splits simultaneously.
    """
    n = Dphi.shape[0]
    A = group_a_mat.astype(np.float64)
    nA = A.sum(axis=0)
    nB = n - nA

    Dsum = Dphi.sum(axis=1)
    total = Dphi.sum()

    DA = Dphi @ A
    sumAA = np.einsum("ib,ib->b", A, DA)
    aDotDsum = A.T @ Dsum
    crossSum = aDotDsum - sumAA
    sumBB = total - 2 * aDotDsum + sumAA

    return (nA * nB / (nA + nB)) * (
        2.0 / (nA * nB) * crossSum - (1.0 / nA ** 2) * sumAA - (1.0 / nB ** 2) * sumBB
    )


class FastCramerTest:
    """Drop-in replacement for CramerTest (multivariate_tests_from_R.py)."""

    def __init__(self, values1: np.ndarray, values2: np.ndarray) -> None:
        self.x = values1
        self.y = values2

    def conduct_test(self, nboot: int = 1000, kernel: str = "phiLog"):
        assert kernel == "phiLog", "only the phiLog kernel is implemented"
        x, y = np.asarray(self.x), np.asarray(self.y)
        n, m = len(x), len(y)
        pooled = np.vstack([x, y])
        Dphi = phi_log(cdist(pooled, pooled))

        rng = np.random.default_rng()
        group_mat = np.zeros((n + m, nboot + 1), dtype=bool)
        group_mat[:n, 0] = True
        for b in range(nboot):
            perm = rng.permutation(n + m)
            group_mat[perm[:n], b + 1] = True

        T = energy_stat_batch(Dphi, group_mat)
        statistic, T_perm = T[0], T[1:]
        pvalue = float(np.mean(T_perm >= statistic))
        return pvalue, float(statistic)


def fast_cramer_scan(X, window_size, shift, nboot=200, seed=0, phi=phi_log):
    """Drop-in replacement for ChangeDetector(test_name="CramerTest").fit(...)."""
    rng = np.random.default_rng(seed)
    n_rows, _ = X.shape
    w = window_size

    Dphi_full = phi(cdist(X, X, metric="euclidean"))

    ids, stats, pvals = [], [], []
    n = 2 * w
    idx_a_fixed = np.zeros(n, dtype=bool)
    idx_a_fixed[:w] = True

    for ind in range(0, n_rows - 2 * w, shift):
        sl = slice(ind, ind + 2 * w)
        Dphi = Dphi_full[sl, sl]

        group_mat = np.empty((n, nboot + 1), dtype=bool)
        group_mat[:, 0] = idx_a_fixed
        for b in range(nboot):
            perm = rng.permutation(n)
            g = np.zeros(n, dtype=bool)
            g[perm[:w]] = True
            group_mat[:, b + 1] = g

        T_all = energy_stat_batch(Dphi, group_mat)
        T_obs, T_perm = T_all[0], T_all[1:]
        pval = float(np.mean(T_perm >= T_obs))

        ids.append(ind + w)
        stats.append(T_obs)
        pvals.append(pval)

    return pd.DataFrame({
        "id": ids,
        "window1_start": [i - w for i in ids],
        "window2_end": [i + w for i in ids],
        "statistic": stats,
        "pvalue": pvals,
    })
