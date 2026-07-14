#!/usr/bin/env python3
"""Student-t ECF-vs-MIDAST known-k=1 grid.

d=2 and d=10 use different parameter sets and are kept as separate pipelines
below: d=2 uses the shared ECF/MIDAST engines on a 9x19 (nu2 x rho2) grid; d=10
uses a smaller 5x7 grid with its own window/stride, reflecting the higher cost
of running MIDAST at d=10.

Usage:
    python run_studentt.py --dim 2
    python run_studentt.py --dim 10
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
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from common.data_simulators import generate_student_t_segment
from common.algorithm12 import algorithm_1_window, k_rule_of_thumb
from ecf.known_k import ECF, mae_known_k
from midast import engine as mdst


def make_studentt(n, n_star, rho1, rho2, nu1, nu2, d):
    seg1 = generate_student_t_segment(nu=nu1, rho=rho1, n=n_star, p=d)
    seg2 = generate_student_t_segment(nu=nu2, rho=rho2, n=n - n_star, p=d)
    return np.vstack([seg1, seg2]), n_star


# ---------------------------------------------------------------- d = 2 ----
N = 1000
N_STAR = 500
WINDOW, GAP, SCAN_STEP, SMOOTH = 150, 10, 5, 5
N_CPS = 1

NU1_D2, RHO1_D2 = 2.0, 0.5
NU2_GRID_D2 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RHO2_GRID_D2 = np.round(np.arange(-0.9, 0.91, 0.1), 1)

MIDAST_SHIFT_D2 = 10
MIDAST_ALPHA = 0.05
TARGET_POWER = 0.9
W_GRID_D2 = (50, 75, 100, 150, 200)
CALIB_M = int(os.getenv("CALIB_M", 100))


def _ecf_cell_worker_d2(args):
    nu2, r2, n_trials, base_seed = args
    ecf = ECF(d=2, window=WINDOW, gap=GAP, scan_step=SCAN_STEP, smooth=SMOOTH)
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_studentt(N, N_STAR, RHO1_D2, r2, NU1_D2, nu2, 2)
        t0 = time.perf_counter()
        preds = ecf.detect_known_k(X, n_cps=N_CPS)
        dt = time.perf_counter() - t0
        rows.append({"nu2": nu2, "rho2": r2, "MAE": mae_known_k([cp], preds), "time_s": dt})
    return rows


def _run_ecf_d2(n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", "studentt_d2", "ecf")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))
    partial = os.path.join(outdir, "raw_partial.csv")

    cells = [(nu2, r2, n_trials, idx * n_trials)
             for idx, (nu2, r2) in enumerate((n, r) for n in NU2_GRID_D2 for r in RHO2_GRID_D2)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["nu2", "rho2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[ECF d=2] {len(NU2_GRID_D2)}x{len(RHO2_GRID_D2)} grid, {n_trials} trials/cell, "
          f"resume: {len(done_keys)} done, {len(todo)} to go", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as pool:
            futs = {pool.submit(_ecf_cell_worker_d2, c): c for c in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    print(f"[ECF d=2] done: {len(df)} rows, mean MAE={df['MAE'].mean():.2f}", flush=True)
    return df


def _midast_cell_worker_d2(args):
    nu2, r2, w, shift_group, n_trials, base_seed = args
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_studentt(N, N_STAR, RHO1_D2, r2, NU1_D2, nu2, 2)
        t0 = time.perf_counter()
        pred = mdst.detect_ks(X, w, shift_group, MIDAST_SHIFT_D2, 2, MIDAST_ALPHA)
        dt = time.perf_counter() - t0
        mae = float(abs(pred - cp)) if pred is not None else np.nan
        rows.append({"nu2": nu2, "rho2": r2, "MAE": mae, "time_s": dt})
    return rows


def _calib_cell_d2(args):
    nu2, r2, base_seed, w_grid, calib_m = args
    powers = {}
    for w in w_grid:
        rejects = 0
        for _ in range(calib_m):
            v1 = generate_student_t_segment(nu=NU1_D2, rho=RHO1_D2, n=w, p=2)
            v2 = generate_student_t_segment(nu=nu2, rho=r2, n=w, p=2)
            _, pval = mdst.ks_two_sample(v1, v2, 2, MIDAST_ALPHA)
            if pval <= MIDAST_ALPHA:
                rejects += 1
        powers[int(w)] = rejects / calib_m
    chosen = next((w for w in w_grid if powers[int(w)] >= TARGET_POWER), max(w_grid))
    return {"nu2": nu2, "rho2": r2, "w_opt": int(chosen), "powers": powers}


def _calibrate_d2(outdir, n_workers):
    cache = os.path.join(outdir, "calibration.json")
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        return c["W_star"], c["k"]

    pairs = [(nu2, r2) for nu2 in NU2_GRID_D2 for r2 in RHO2_GRID_D2 if not (nu2 == NU1_D2 and r2 == RHO1_D2)]
    cells = [(nu2, r2, i * 9973 + 17, W_GRID_D2, CALIB_M) for i, (nu2, r2) in enumerate(pairs)]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_calib_cell_d2, c): c for c in cells}
        for fut in as_completed(futs):
            results.append(fut.result())
    w_opts = [r["w_opt"] for r in results]
    W_star = int(max(w_opts))
    k = k_rule_of_thumb(W_star, MIDAST_SHIFT_D2)
    c = {"W_star": W_star, "k": k, "s": MIDAST_SHIFT_D2, "per_cell": results}
    with open(cache, "w") as f:
        json.dump(c, f, indent=2)
    return W_star, k


def _run_midast_d2(n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", "studentt_d2", "midast")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 1000))

    W_star, k = _calibrate_d2(outdir, n_workers)
    shift_group = mdst.shift_group_from_k(k, W_star, MIDAST_SHIFT_D2)

    partial = os.path.join(outdir, "raw_partial.csv")
    cells = [(nu2, r2, W_star, shift_group, n_trials, idx * n_trials)
             for idx, (nu2, r2) in enumerate((n, r) for n in NU2_GRID_D2 for r in RHO2_GRID_D2)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["nu2", "rho2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[MIDAST d=2] W*={W_star} k={k} shift_group={shift_group}, "
          f"resume: {len(done_keys)} done, {len(todo)} to go", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as pool:
            futs = {pool.submit(_midast_cell_worker_d2, c): c for c in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    print(f"[MIDAST d=2] done: {len(df)} rows", flush=True)
    return df


# --------------------------------------------------------------- d = 10 ----
DIM_D10 = 10
N_D10 = 1000
WINDOW_D10, GAP_D10, SCAN_STEP_D10, SMOOTH_D10 = 150, 10, 10, 5
M_FREQ_D10 = 256
SCALES_D10 = (0.5, 1.0, 2.0)
SEED_D10 = 0

MIDAST_SHIFT_D10 = 5
RHO1_D10, NU1_D10 = 0.5, 5.0
NU2_GRID_D10 = [2.0, 3.0, 5.0, 8.0, 12.0]
RHO2_GRID_D10 = np.round(np.arange(-0.9, 1.0, 0.3), 1)


class _FixedECF:
    """Float64, non-vectorized ECF detector used for the d=10 grid."""

    def __init__(self, d, M=M_FREQ_D10, scales=SCALES_D10, seed=SEED_D10, feature="cos"):
        rng = np.random.default_rng(seed)
        self.U = rng.standard_normal((M, d)) / np.sqrt(d)
        self.scales = scales
        self.feature = feature

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        return (X - med) / scale

    def fingerprint(self, win):
        feats = []
        for s in self.scales:
            S = win @ (self.U * s).T
            feats.append(np.cos(S).mean(axis=0))
            if self.feature == "cossin":
                feats.append(np.sin(S).mean(axis=0))
        z = np.concatenate(feats)
        return z / (np.linalg.norm(z) + 1e-12)

    def score_series(self, X, L=WINDOW_D10, gap=GAP_D10, step=SCAN_STEP_D10, smooth=SMOOTH_D10):
        Xs = self._robust_standardize(X)
        n = len(Xs)
        idx = np.arange(L, n - L - gap, step)
        scores = np.empty(len(idx))
        for i, t in enumerate(idx):
            zp = self.fingerprint(Xs[t - L:t])
            zf = self.fingerprint(Xs[t + gap:t + L + gap])
            scores[i] = 1.0 - float(zp @ zf)
        if smooth > 1 and len(scores) >= smooth:
            scores = np.convolve(scores, np.ones(smooth) / smooth, mode="same")
        return idx, scores

    @staticmethod
    def _extract(idx, scores, mode):
        if len(scores) == 0:
            return None
        if mode == "argmax":
            return int(idx[int(np.argmax(scores))])
        peaks, props = find_peaks(scores, prominence=1e-6, distance=WINDOW_D10 // SCAN_STEP_D10)
        if len(peaks) == 0:
            return int(idx[int(np.argmax(scores))])
        best = peaks[int(np.argmax(props["prominences"]))]
        return int(idx[best])

    def detect(self, X, mode="argmax"):
        idx, scores = self.score_series(X)
        return self._extract(idx, scores, mode)


def _select_w_k_d10():
    w = algorithm_1_window(
        test_name="KSTest", dist="student_t", dim=DIM_D10,
        rho_pre=0.5, rho_post=0.0, tail_pre=3.0, tail_post=8.0,
        M=200, target_power=0.9, c=0.05,
    )
    return int(w), int(k_rule_of_thumb(w, MIDAST_SHIFT_D10))


def _run_midast_d10(X):
    from vendored_midast.multivariate_statistical_test_method import ChangeDetector
    w, k = _select_w_k_d10()
    s = MIDAST_SHIFT_D10
    shift_group = max(1, int(k * w / 100 * s))
    det = ChangeDetector(test_name="KSTest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_df = det.fit(X, window_size=w, shift=s)
    cps = det.analyze_results(res_df, output_type="np.array", alpha=0.05,
                              shift_group=shift_group, max_no_changes=1, based_on="statistic")
    if cps is None or np.asarray(cps).size == 0:
        return None
    return int(np.asarray(cps)[0])


def _mae(true_cp, pred):
    return np.nan if pred is None else abs(pred - true_cp)


def _cell_worker_d10(args):
    nu2, r2, n_trials, base_seed = args
    ecf = _FixedECF(d=DIM_D10)
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_studentt(N_D10, N_D10 // 2, RHO1_D10, r2, NU1_D10, nu2, DIM_D10)
        idx, scores = ecf.score_series(X)
        rows.append({"Method": "ECF[argmax]", "rho2": r2, "nu2": nu2,
                     "MAE": _mae(cp, ecf._extract(idx, scores, "argmax")), "time_s": np.nan})
        rows.append({"Method": "ECF[peaks]", "rho2": r2, "nu2": nu2,
                     "MAE": _mae(cp, ecf._extract(idx, scores, "peaks")), "time_s": np.nan})
        t0 = time.time()
        pred = _run_midast_d10(X)
        rows.append({"Method": "MIDAST[KS]", "rho2": r2, "nu2": nu2,
                     "MAE": _mae(cp, pred), "time_s": time.time() - t0})
    return rows


def run_studentt_d10(n_workers):
    outdir = os.path.join(ROOT, "results", "known_k", "studentt_d10")
    os.makedirs(outdir, exist_ok=True)
    n_trials = int(os.getenv("TRIALS_A", 30))
    partial = os.path.join(outdir, "raw_partial.csv")

    cells = [(nu2, r2, n_trials, idx * n_trials)
             for idx, (nu2, r2) in enumerate((nu, r) for nu in NU2_GRID_D10 for r in RHO2_GRID_D10)]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["nu2", "rho2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]
    print(f"[d=10, s={MIDAST_SHIFT_D10}] {len(NU2_GRID_D10)}x{len(RHO2_GRID_D10)} grid, "
          f"{n_trials} trials/cell, resume: {len(done_keys)} done, {len(todo)} to go", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as pool:
            futs = {pool.submit(_cell_worker_d10, c): c for c in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    print(f"[d=10] done: {len(df)} rows", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 10], required=True)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    ap.add_argument("--skip-midast", action="store_true")
    args = ap.parse_args()

    if args.dim == 2:
        print("=== Student-t, d=2 ===", flush=True)
        _run_ecf_d2(args.workers)
        if not args.skip_midast:
            _run_midast_d2(args.workers)
    else:
        print("=== Student-t, d=10 ===", flush=True)
        run_studentt_d10(args.workers)


if __name__ == "__main__":
    main()
