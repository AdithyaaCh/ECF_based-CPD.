#!/usr/bin/env python3
"""Lomax (Pareto Type II) ECF-vs-MIDAST known-k=1 grid, d in {2, 10}.

Each coordinate is an independent np.random.pareto(alpha) draw; alpha alone
controls tail heaviness. W_GRID is wider than for sub-Gaussian/Student-t
(extends to 300, 400) since Lomax's heavier tails need larger windows to
reach the target calibration power in the hardest cells.

Lomax is the one family where the sine (imaginary-CF) feature helps rather
than hurts, so ECF is run with feature="cossin" here (see the ablation study).

Usage:
    python run_pareto.py --dim 2
    python run_pareto.py --dim 10
"""
import os
import sys
import json
import time
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from ecf.known_k import ECF, mae_known_k
from midast import engine as mdst
from common.algorithm12 import k_rule_of_thumb

N = 1000
N_STAR = 500
WINDOW, GAP, SCAN_STEP, SMOOTH = 150, 10, 5, 5
N_CPS = 1

ALPHA1_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
ALPHA2_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

MIDAST_SHIFT = 10
MIDAST_ALPHA = 0.05
TARGET_POWER = 0.9
W_GRID = (50, 75, 100, 150, 200, 300, 400)
CALIB_M = int(os.getenv("CALIB_M", 100))


def make_lomax(n, n_star, alpha1, alpha2, d):
    seg1 = np.random.pareto(alpha1, size=(n_star, d))
    seg2 = np.random.pareto(alpha2, size=(n - n_star, d))
    return np.vstack([seg1, seg2]), n_star


def _ecf_cell_worker(args):
    a1, a2, n_trials, base_seed, dim = args
    ecf = ECF(d=dim, window=WINDOW, gap=GAP, scan_step=SCAN_STEP, smooth=SMOOTH, feature="cossin")
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_lomax(N, N_STAR, a1, a2, dim)
        t0 = time.perf_counter()
        preds = ecf.detect_known_k(X, n_cps=N_CPS)
        dt = time.perf_counter() - t0
        rows.append({"alpha1": a1, "alpha2": a2, "MAE": mae_known_k([cp], preds), "time_s": dt})
    return rows


def run_ecf(dim, n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", f"lomax_d{dim}", "ecf")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))
    partial = os.path.join(outdir, "raw_partial.csv")

    cells = [(a1, a2, n_trials, idx * n_trials, dim)
             for idx, (a1, a2) in enumerate((x, y) for x in ALPHA1_GRID for y in ALPHA2_GRID)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["alpha1", "alpha2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[ECF d={dim}] {len(ALPHA1_GRID)}x{len(ALPHA2_GRID)} grid, {n_trials} trials/cell, "
          f"resume: {len(done_keys)} done, {len(todo)} to go", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as pool:
            futs = {pool.submit(_ecf_cell_worker, c): c for c in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    print(f"[ECF d={dim}] done: {len(df)} rows, mean MAE={df['MAE'].mean():.2f}", flush=True)
    return df


def _midast_cell_worker(args):
    a1, a2, w, shift_group, n_trials, base_seed, dim = args
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_lomax(N, N_STAR, a1, a2, dim)
        t0 = time.perf_counter()
        pred = mdst.detect_ks(X, w, shift_group, MIDAST_SHIFT, dim, MIDAST_ALPHA)
        dt = time.perf_counter() - t0
        mae = float(abs(pred - cp)) if pred is not None else np.nan
        rows.append({"alpha1": a1, "alpha2": a2, "MAE": mae, "time_s": dt})
    return rows


def _calib_cell(args):
    a1, a2, base_seed, dim, w_grid, calib_m, alpha = args
    rng = np.random.default_rng(base_seed)
    powers = {}
    for w in w_grid:
        rejects = 0
        for _ in range(calib_m):
            v1 = rng.pareto(a1, size=(w, dim))
            v2 = rng.pareto(a2, size=(w, dim))
            _, pval = mdst.ks_two_sample(v1, v2, dim, alpha)
            if pval <= alpha:
                rejects += 1
        powers[int(w)] = rejects / calib_m
    chosen = next((w for w in w_grid if powers[int(w)] >= TARGET_POWER), max(w_grid))
    return {"alpha1": a1, "alpha2": a2, "w_opt": int(chosen), "powers": powers}


def _calibrate(outdir, pairs, dim, w_grid, calib_m, midast_shift, n_workers, alpha):
    """Lomax's pre-change segment depends on alpha1, which varies per cell
    (unlike sub-Gaussian/Student-t), so calibration is done directly here."""
    cache = os.path.join(outdir, "calibration.json")
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        return c["W_star"], c["k"], c

    cells = [(a1, a2, i * 9973 + 17, dim, w_grid, calib_m, alpha) for i, (a1, a2) in enumerate(pairs)]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_calib_cell, c): c for c in cells}
        for fut in as_completed(futs):
            results.append(fut.result())

    w_opts = [r["w_opt"] for r in results]
    W_star = int(max(w_opts))
    k = k_rule_of_thumb(W_star, midast_shift)
    c = {"W_star": W_star, "k": k, "s": midast_shift, "target_power": TARGET_POWER,
         "calib_M": calib_m, "w_grid": list(w_grid), "per_cell": results}
    with open(cache, "w") as f:
        json.dump(c, f, indent=2)
    return W_star, k, c


def run_midast(dim, n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", f"lomax_d{dim}", "midast")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))

    pairs = [(a1, a2) for a1 in ALPHA1_GRID for a2 in ALPHA2_GRID if a1 != a2]
    W_star, k, _ = _calibrate(outdir, pairs, dim, W_GRID, CALIB_M, MIDAST_SHIFT, n_workers, MIDAST_ALPHA)
    shift_group = mdst.shift_group_from_k(k, W_star, MIDAST_SHIFT)

    partial = os.path.join(outdir, "raw_partial.csv")
    cells = [(a1, a2, W_star, shift_group, n_trials, idx * n_trials, dim)
             for idx, (a1, a2) in enumerate((x, y) for x in ALPHA1_GRID for y in ALPHA2_GRID)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["alpha1", "alpha2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[MIDAST d={dim}] W*={W_star} k={k} shift_group={shift_group}, "
          f"resume: {len(done_keys)} done, {len(todo)} to go", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as pool:
            futs = {pool.submit(_midast_cell_worker, c): c for c in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    print(f"[MIDAST d={dim}] done: {len(df)} rows", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 10], required=True)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    ap.add_argument("--skip-ecf", action="store_true")
    ap.add_argument("--skip-midast", action="store_true")
    args = ap.parse_args()

    print(f"=== Lomax / Pareto, d={args.dim} ===", flush=True)
    if not args.skip_ecf:
        run_ecf(args.dim, args.workers)
    if not args.skip_midast:
        run_midast(args.dim, args.workers)


if __name__ == "__main__":
    main()
