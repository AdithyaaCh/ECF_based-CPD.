import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from vendored_midast.multivariate_statistical_test_method import ChangeDetector
from vendored_midast.ndtest import ks2d2s
from vendored_midast.ks_2samp import ks_2samp
from common.algorithm12 import k_rule_of_thumb

MIDAST_ALPHA = 0.05
TARGET_POWER = 0.9


def ks_two_sample(v1, v2, dim, alpha=MIDAST_ALPHA):
    if dim == 2:
        x1, y1 = v1.T
        x2, y2 = v2.T
        pval, stat = ks2d2s(x1, y1, x2, y2, extra=True)
        return float(stat), float(pval)
    stat, _, _, pval = ks_2samp(x_val=v1, y_val=v2, alpha=alpha)
    return float(stat), float(pval)


def _calib_cell(args, gen_pre, gen_post, dim, w_grid, calib_m, alpha):
    key1, key2, base_seed = args
    rng = np.random.default_rng(base_seed)
    powers = {}
    for w in w_grid:
        rejects = 0
        for _ in range(calib_m):
            v1 = gen_pre(rng, w, dim)
            v2 = gen_post(rng, key2, w, dim)
            _, pval = ks_two_sample(v1, v2, dim, alpha)
            if pval <= alpha:
                rejects += 1
        powers[int(w)] = rejects / calib_m
    chosen = next((w for w in w_grid if powers[int(w)] >= TARGET_POWER), max(w_grid))
    return {"key1": key1, "key2": key2, "w_opt": int(chosen), "powers": powers}


def calibrate(outdir, pairs, gen_pre, gen_post, dim, w_grid, calib_m,
              midast_shift, n_workers, alpha=MIDAST_ALPHA):
    """Returns (W_star, k, calib_dict); cached to outdir/calibration.json."""
    cache = os.path.join(outdir, "calibration.json")
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        return c["W_star"], c["k"], c

    cells = [(k1, k2, i * 9973 + 17) for i, (k1, k2) in enumerate(pairs)]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_calib_cell, c, gen_pre, gen_post, dim, w_grid, calib_m, alpha): c
                for c in cells}
        for fut in as_completed(futs):
            results.append(fut.result())

    w_opts = [r["w_opt"] for r in results]
    W_star = int(max(w_opts))
    k = k_rule_of_thumb(W_star, midast_shift)
    c = {"W_star": W_star, "k": k, "s": midast_shift,
         "target_power": TARGET_POWER, "calib_M": calib_m,
         "w_grid": list(w_grid), "per_cell": results}
    with open(cache, "w") as f:
        json.dump(c, f, indent=2)
    return W_star, k, c


def shift_group_from_k(k, W_star, midast_shift):
    return max(1, int(k * W_star / 100 * midast_shift))


def detect_ks(X, w, shift_group, midast_shift, dim, alpha=MIDAST_ALPHA):
    det = ChangeDetector(test_name="KSTest")
    res = det.fit(X, window_size=w, shift=midast_shift)
    cps = det.analyze_results(res, output_type="np.array", alpha=alpha,
                              shift_group=shift_group, max_no_changes=1,
                              based_on="statistic")
    if cps is None or len(cps) == 0:
        return None
    return int(np.sort(np.asarray(cps).astype(int))[0])
