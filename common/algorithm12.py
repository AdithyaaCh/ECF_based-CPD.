"""

Algorithm 1
-----------
For each candidate w, simulate `M` two-segment series (one regime each side)
under a representative pre/post parameter swap, run the chosen two-sample test
once per series, and call the test "powered" at level c=0.05 if p <= c. Pick
the smallest w whose empirical power >= target_power.

Algorithm 2
-----------
For each candidate k, run MIDAST end-to-end on `M` synthetic single-CP series
of length 2*N* (so the change-point lies at position N*) and compute the median
absolute positional error. Pick the smallest k whose median error is below
`tol`.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from common import data_simulators as ds
from vendored_midast import ChangeDetector



@dataclass(frozen=True)
class CalibKey:
    test_name: str
    dist: str
    dim: int
    rho_pre: float
    rho_post: float
    nu_or_alpha_pre: float
    nu_or_alpha_post: float
    M: int
    target_power: float
    c: float


def _cache_path(root: str, key: CalibKey, kind: str) -> str:
    os.makedirs(root, exist_ok=True)
    fname = f"{kind}_{key.test_name}_{key.dist}_d{key.dim}_M{key.M}.json"
    return os.path.join(root, fname)


def _segment(dist: str, p: int, n: int, rho: float, tail: float) -> np.ndarray:
    if dist == "student_t":
        return ds.generate_student_t_segment(nu=tail, rho=rho, n=n, p=p)
    elif dist == "sub_gaussian":
        return ds.generate_subgaussian_segment(alpha=tail, rho=rho, n=n, p=p)
    raise ValueError(f"unknown dist {dist}")


def algorithm_1_window(
    test_name: str,
    dist: str,
    dim: int,
    rho_pre: float,
    rho_post: float,
    tail_pre: float,
    tail_post: float,
    *,
    w_grid: Iterable[int] = (50, 75, 100, 150, 200, 300, 400),
    M: int = 200,
    target_power: float = 0.9,
    c: float = 0.05,
    cache_dir: Optional[str] = "checkpoints/calib",
    verbose: bool = False,
) -> int:
    """Return the smallest w in `w_grid` whose empirical power >= target_power."""
    key = CalibKey(
        test_name=test_name,
        dist=dist,
        dim=dim,
        rho_pre=rho_pre,
        rho_post=rho_post,
        nu_or_alpha_pre=tail_pre,
        nu_or_alpha_post=tail_post,
        M=M,
        target_power=target_power,
        c=c,
    )
    if cache_dir:
        cp = _cache_path(cache_dir, key, "alg1")
        if os.path.exists(cp):
            with open(cp) as fh:
                return int(json.load(fh)["w"])

    powers: Dict[int, float] = {}
    for w in w_grid:
        rejects = 0
        for _ in range(M):
            v1 = _segment(dist, dim, w, rho_pre, tail_pre)
            v2 = _segment(dist, dim, w, rho_post, tail_post)
            stat, pval = _run_two_sample(test_name, v1, v2, dim)
            if pval <= c:
                rejects += 1
        powers[w] = rejects / M
        if verbose:
            print(f"  alg1 w={w}: power={powers[w]:.3f}")

    chosen = next((w for w in w_grid if powers[w] >= target_power), max(w_grid))
    if cache_dir:
        with open(cp, "w") as fh:
            json.dump({"w": chosen, "powers": powers, "key": asdict(key)}, fh)
    return int(chosen)


def algorithm_2_k(
    test_name: str,
    dist: str,
    dim: int,
    w: int,
    s: int,
    rho_pre: float,
    rho_post: float,
    tail_pre: float,
    tail_post: float,
    *,
    k_grid: Iterable[int] = (1, 2, 3, 5, 8, 13),
    M: int = 100,
    tol: int = 50,
    c: float = 0.05,
    cache_dir: Optional[str] = "checkpoints/calib",
    verbose: bool = False,
) -> int:
    """Return the smallest k whose median |cp_hat - n*| <= tol."""
    key = CalibKey(
        test_name=test_name,
        dist=dist,
        dim=dim,
        rho_pre=rho_pre,
        rho_post=rho_post,
        nu_or_alpha_pre=tail_pre,
        nu_or_alpha_post=tail_post,
        M=M,
        target_power=0.0,
        c=c,
    )
    if cache_dir:
        cp = _cache_path(cache_dir, key, f"alg2_w{w}_s{s}_tol{tol}")
        if os.path.exists(cp):
            with open(cp) as fh:
                return int(json.load(fh)["k"])

    median_err: Dict[int, float] = {}
    for k in k_grid:
        errs: List[float] = []
        # Algorithm 2 (paper pseudocode): "use MIDAST with selected w, s, k" --
        # k is passed directly as the grouping factor, no rescaling by w/100*s.
        # (That rescaling appeared in an earlier version of this function but
        # doesn't exist in the paper; it conflated the rule-of-thumb estimate
        # of k with how a chosen k is actually *used*.)
        shift_group = max(1, int(k))
        for _ in range(M):
            n_star = 4 * w  # generous so both halves can fit
            n_total = 2 * n_star
            seg1 = _segment(dist, dim, n_star, rho_pre, tail_pre)
            seg2 = _segment(dist, dim, n_total - n_star, rho_post, tail_post)
            X = np.vstack([seg1, seg2])

            det = ChangeDetector(test_name=test_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_df = det.fit(X, window_size=w, shift=s)
                cps = det.analyze_results(
                    res_df,
                    output_type="np.array",
                    alpha=c,
                    shift_group=shift_group,
                    based_on="statistic",
                )
            except Exception:
                cps = None
            if cps is None or len(cps) == 0:
                errs.append(float(n_total))  
            else:
                errs.append(float(np.min(np.abs(np.asarray(cps) - n_star))))
        median_err[k] = float(np.median(errs))
        if verbose:
            print(f"  alg2 k={k}: median_err={median_err[k]:.1f}")

    chosen = next((k for k in k_grid if median_err[k] <= tol), max(k_grid))
    if cache_dir:
        with open(cp, "w") as fh:
            json.dump({"k": chosen, "median_err": median_err, "key": asdict(key)}, fh)
    return int(chosen)



def k_rule_of_thumb(w: int, s: int) -> int:
    return max(1, int(round(w / (100 * s))))



def _run_two_sample(test_name: str, v1: np.ndarray, v2: np.ndarray, dim: int) -> Tuple[float, float]:
    """Single-call wrapper around the vendored MIDAST tests, returns (statistic, pvalue).

    We call the test functions directly instead of going through `ChangeDetector.test_in_window`
    because that loops over `range(0, n_rows - 2*window_size, shift)` which is empty when
    `n_rows == 2*window_size`. The dispatch here mirrors `test_in_window` byte-for-byte.
    """
    from vendored_midast.ks_2samp import ks_2samp
    from vendored_midast.ndtest import ks2d2s

    if test_name in ("KSTest", "KSTest_DKW"):
        if dim == 2 and test_name == "KSTest":
            x1, y1 = v1.T
            x2, y2 = v2.T
            pval, stat = ks2d2s(x1, y1, x2, y2, extra=True)
            return float(stat), float(pval)
        stat, _, _, pval = ks_2samp(v1, v2, alpha=0.05)
        return float(stat), float(pval)

    if test_name == "MMDTest":

        from sklearn.metrics.pairwise import rbf_kernel

        gamma = 1.0
        n, m = v1.shape[0], v2.shape[0]
        XX = rbf_kernel(v1, v1, gamma)
        YY = rbf_kernel(v2, v2, gamma)
        XY = rbf_kernel(v1, v2, gamma)
        true_stat = (
            (XX.sum() - np.trace(XX)) / (n * (n - 1))
            + (YY.sum() - np.trace(YY)) / (m * (m - 1))
            - 2 * XY.sum() / (n * m)
        )
        pooled = np.vstack([v1, v2])
        n_perm = 99
        ge = 0
        for _ in range(n_perm):
            np.random.shuffle(pooled)
            a, b = pooled[:n], pooled[n:]
            AA = rbf_kernel(a, a, gamma)
            BB = rbf_kernel(b, b, gamma)
            AB = rbf_kernel(a, b, gamma)
            fake = (
                (AA.sum() - np.trace(AA)) / (n * (n - 1))
                + (BB.sum() - np.trace(BB)) / (m * (m - 1))
                - 2 * AB.sum() / (n * m)
            )
            if fake >= true_stat:
                ge += 1
        return float(true_stat), float((ge + 1) / (n_perm + 1))

    raise ValueError(f"unsupported test_name {test_name} in calibration path")
