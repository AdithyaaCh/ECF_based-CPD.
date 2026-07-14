#!/usr/bin/env python3
"""MIDAST[KS] vs ECF-Divisive vs e-Divisive, unknown-k, on sub-Gaussian
joint-change data (alpha 1.9<->1.6 and rho 0.6<->-0.6 together), N=1000,
k in {0, 1, 2, 3} evenly spaced change-points (k=0 = stationary).

Usage:
    python three_way_comparison.py --dim 2
    python three_way_comparison.py --dim 10
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from common.data_simulators import generate_subgaussian_segment
from ecf.divisive import ECFCore, ecf_divisive
from baselines.e_divisive import e_divisive

HI = (1.9, 0.6)
LO = (1.6, -0.6)
N = 1000
W = 200
S = 1
SG = 20
N_TRIALS = 50


def make_joint(k, n=N, d=2):
    if k == 0:
        a, r = HI
        return generate_subgaussian_segment(alpha=a, rho=r, n=n, p=d), []
    cps = [int(round(n * (i + 1) / (k + 1))) for i in range(k)]
    bounds = [0] + cps + [n]
    segs = []
    for i in range(len(bounds) - 1):
        a, r = HI if i % 2 == 0 else LO
        segs.append(generate_subgaussian_segment(alpha=a, rho=r, n=bounds[i + 1] - bounds[i], p=d))
    return np.vstack(segs), cps


def match_mae(true, pred, tol=40):
    tl = list(true)
    errs = []
    for p in sorted(pred):
        if not tl:
            break
        dd = [abs(p - t) for t in tl]
        j = int(np.argmin(dd))
        if dd[j] <= tol:
            errs.append(dd[j])
            tl.pop(j)
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def _trial(args):
    method, k, trial, dim = args
    from vendored_midast.multivariate_statistical_test_method import ChangeDetector

    np.random.seed(7000 * k + trial)
    X, true = make_joint(k, d=dim)

    t0 = time.perf_counter()
    if method == "midast":
        det = ChangeDetector(test_name="KSTest")
        res = det.fit(X, window_size=W, shift=S)
        cps = det.analyze_results(res, output_type="np.array", alpha=0.05,
                                  shift_group=SG, max_no_changes=None, based_on="statistic")
        pred = [] if cps is None or len(cps) == 0 else sorted(int(c) for c in np.asarray(cps))
    elif method == "ecf":
        core = ECFCore(d=dim)
        feat = core.feature_matrix(core.robust_standardize(X))
        rng = np.random.default_rng(9000 + 13 * trial + k)
        pred = ecf_divisive(feat, rng)
    else:
        pred = e_divisive(X)
    dt = time.perf_counter() - t0

    mae, tp = match_mae(true, pred) if k > 0 else (np.nan, 0)
    return {"method": method, "true_k": k, "trial": trial, "k_hat": len(pred), "mae": mae, "time_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 10], required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "results", "unknown_k", f"three_way_d{args.dim}")
    os.makedirs(outdir, exist_ok=True)
    partial_path = os.path.join(outdir, "raw_partial.csv")

    done_keys = set()
    if os.path.exists(partial_path):
        df = pd.read_csv(partial_path)
        done_keys = set(map(tuple, df[["method", "true_k", "trial"]].values.tolist()))
        print(f"[resume] {len(done_keys)} trials already done", flush=True)

    tasks = [(m, k, t, args.dim) for m in ["midast", "ecf", "edivisive"] for k in [0, 1, 2, 3]
             for t in range(args.trials) if (m, k, t) not in done_keys]
    total = args.trials * 4 * 3
    print(f"Running {len(tasks)} remaining trials, dim={args.dim}, workers={args.workers}", flush=True)

    done_count = len(done_keys)
    buffer = []

    def _flush():
        nonlocal buffer
        if buffer:
            pd.DataFrame(buffer).to_csv(partial_path, mode="a", header=not os.path.exists(partial_path), index=False)
            buffer = []

    if args.workers <= 1:
        for task in tasks:
            buffer.append(_trial(task))
            done_count += 1
            if len(buffer) >= 5:
                _flush()
            if done_count % 20 == 0:
                print(f"  {done_count}/{total} done", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_trial, t): t for t in tasks}
            for fut in as_completed(futs):
                buffer.append(fut.result())
                done_count += 1
                if len(buffer) >= 10:
                    _flush()
                if done_count % 25 == 0:
                    print(f"  {done_count}/{total} done", flush=True)
    _flush()

    df = pd.read_csv(partial_path)
    print(f"\nFINAL: N={N} dim={args.dim} MIDAST(w={W},s={S},sg={SG}), unknown-k, {args.trials} trials/k", flush=True)
    for method in ["midast", "ecf", "edivisive"]:
        for k in [0, 1, 2, 3]:
            s = df[(df.method == method) & (df.true_k == k)]
            if len(s) == 0:
                continue
            exact_pct = 100 * (s.k_hat == k).mean()
            print(f"  {method:10s} k={k}: exact-k={exact_pct:5.0f}%  mean_khat={s.k_hat.mean():5.2f}  "
                  f"MAE={s.mae.mean():6.1f}  time={s.time_s.mean()*1000:8.1f}ms", flush=True)


if __name__ == "__main__":
    main()
