#!/usr/bin/env python3
"""Sub-Gaussian ECF-vs-MIDAST known-k=1 grid, d in {2, 10}.

Usage:
    python run_subgaussian.py --dim 2
    python run_subgaussian.py --dim 10
"""
import os
import sys
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

from common.data_simulators import generate_subgaussian_segment
from ecf.known_k import ECF, mae_known_k
from midast import engine as mdst

N = 1000
N_STAR = 500
WINDOW, GAP, SCAN_STEP, SMOOTH = 150, 10, 5, 5
N_CPS = 1

RHO1 = 0.5
ALPHA1 = 1.9

ALPHA2_GRID = [1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95, 1.98]
RHO2_GRID = np.round(np.arange(-0.9, 0.91, 0.1), 1)

MIDAST_SHIFT = 10
MIDAST_ALPHA = 0.05
TARGET_POWER = 0.9
W_GRID = (50, 75, 100, 150, 200)
CALIB_M = int(os.getenv("CALIB_M", 100))


def make_subgaussian(n, n_star, rho1, rho2, alpha1, alpha2, d):
    seg1 = generate_subgaussian_segment(alpha=alpha1, rho=rho1, n=n_star, p=d)
    seg2 = generate_subgaussian_segment(alpha=alpha2, rho=rho2, n=n - n_star, p=d)
    return np.vstack([seg1, seg2]), n_star


def _gen_pre(rng, w, d):
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    return generate_subgaussian_segment(alpha=ALPHA1, rho=RHO1, n=w, p=d)


def _gen_post(rng, key2, w, d):
    a2, r2 = key2
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    return generate_subgaussian_segment(alpha=a2, rho=r2, n=w, p=d)


def _ecf_cell_worker(args):
    a2, r2, n_trials, base_seed, dim = args
    ecf = ECF(d=dim, window=WINDOW, gap=GAP, scan_step=SCAN_STEP, smooth=SMOOTH)
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_subgaussian(N, N_STAR, RHO1, r2, ALPHA1, a2, dim)
        t0 = time.time()
        preds = ecf.detect_known_k(X, n_cps=N_CPS)
        dt = time.time() - t0
        rows.append({"alpha2": a2, "rho2": r2, "MAE": mae_known_k([cp], preds), "time_s": dt})
    return rows


def run_ecf(dim, n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", f"subgaussian_d{dim}", "ecf")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))
    partial = os.path.join(outdir, "raw_partial.csv")

    cells = [(a2, r2, n_trials, idx * n_trials, dim)
             for idx, (a2, r2) in enumerate((a, r) for a in ALPHA2_GRID for r in RHO2_GRID)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["alpha2", "rho2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[ECF d={dim}] {len(ALPHA2_GRID)}x{len(RHO2_GRID)} grid, {n_trials} trials/cell, "
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
    a2, r2, w, shift_group, n_trials, base_seed, dim = args
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_subgaussian(N, N_STAR, RHO1, r2, ALPHA1, a2, dim)
        t0 = time.perf_counter()
        pred = mdst.detect_ks(X, w, shift_group, MIDAST_SHIFT, dim, MIDAST_ALPHA)
        dt = time.perf_counter() - t0
        mae = float(abs(pred - cp)) if pred is not None else np.nan
        rows.append({"alpha2": a2, "rho2": r2, "MAE": mae, "time_s": dt})
    return rows


def run_midast(dim, n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", f"subgaussian_d{dim}", "midast")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))

    pairs = [(a2, r2) for a2 in ALPHA2_GRID for r2 in RHO2_GRID if not (a2 == ALPHA1 and r2 == RHO1)]
    pairs_for_calib = [(None, (a2, r2)) for a2, r2 in pairs]
    W_star, k, _ = mdst.calibrate(outdir, pairs_for_calib, _gen_pre, _gen_post, dim,
                                   W_GRID, CALIB_M, MIDAST_SHIFT, n_workers, MIDAST_ALPHA)
    shift_group = mdst.shift_group_from_k(k, W_star, MIDAST_SHIFT)

    partial = os.path.join(outdir, "raw_partial.csv")
    cells = [(a2, r2, W_star, shift_group, n_trials, idx * n_trials, dim)
             for idx, (a2, r2) in enumerate((a, r) for a in ALPHA2_GRID for r in RHO2_GRID)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["alpha2", "rho2"]].drop_duplicates().values))
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

    print(f"=== Sub-Gaussian, d={args.dim} ===", flush=True)
    if not args.skip_ecf:
        run_ecf(args.dim, args.workers)
    if not args.skip_midast:
        run_midast(args.dim, args.workers)


if __name__ == "__main__":
    main()
