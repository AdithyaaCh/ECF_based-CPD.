import os
import sys
import time
import tempfile
import subprocess
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from common.data_simulators import generate_subgaussian_segment, generate_student_t_segment, _safe_corr
from common.pareto import generate_lomax_segment
from ecf.divisive import ECFCore, ecf_divisive
from vendored_midast.multivariate_statistical_test_method import ChangeDetector
from midast.cvm_fast import fast_cramer_scan

N = 1000
W, S, SG = 200, 10, 20
TOL = 25
FAMILIES = ["gaussian", "subgaussian", "studentt_nu2", "lomax"]
DIMS = [2, 10]
KVALS = [0, 1, 2, 3]
METHODS = ["ecf", "midast_ks", "midast_cvm", "edivisive"]
HI = {"alpha": 1.9, "rho": 0.6}
LO = {"alpha": 1.6, "rho": -0.6}

OUTDIR = os.path.join(ROOT, "results", "unknown_k", "benchmark")
os.makedirs(OUTDIR, exist_ok=True)
CKPT = os.path.join(OUTDIR, "raw.csv")


def _gauss(rho, n, d):
    return np.random.multivariate_normal(np.zeros(d), _safe_corr(d, rho), size=n, check_valid="ignore")


def seg(family, regime, n, d):
    a, r = regime["alpha"], regime["rho"]
    if family == "gaussian":     return _gauss(r, n, d)
    if family == "subgaussian":  return generate_subgaussian_segment(alpha=a, rho=r, n=n, p=d)
    if family == "studentt_nu2": return generate_student_t_segment(nu=2.0, rho=r, n=n, p=d)
    if family == "lomax":        return generate_lomax_segment(alpha=1.6, rho=r, n=n, p=d, scale=1.0)


def make(family, k, d, seed):
    np.random.seed(seed)
    if k == 0:
        return seg(family, HI, N, d), []
    cps = [int(round(N * (i + 1) / (k + 1))) for i in range(k)]
    bounds = [0] + cps + [N]
    segs = [seg(family, HI if i % 2 == 0 else LO, bounds[i + 1] - bounds[i], d) for i in range(len(bounds) - 1)]
    return np.vstack(segs), cps


_helper = ChangeDetector(test_name="KSTest")


def det_ecf(X, d, seed):
    core = ECFCore(d=d, seed=0)
    feat = core.feature_matrix(core.robust_standardize(X))
    return sorted(ecf_divisive(feat, np.random.default_rng(seed)))


def det_midast(X, test):
    if test == "ks":
        det = ChangeDetector(test_name="KSTest")
        res = det.fit(X, window_size=W, shift=S)
        cps = det.analyze_results(res, output_type="np.array", alpha=0.05,
                                  shift_group=SG, max_no_changes=None, based_on="statistic")
    else:
        rdf = fast_cramer_scan(X, window_size=W, shift=S, nboot=200, seed=0)
        cps = _helper.analyze_results(rdf, output_type="np.array", alpha=0.05,
                                      shift_group=SG, max_no_changes=None, based_on="statistic")
    return [] if cps is None or len(cps) == 0 else sorted(int(c) for c in np.asarray(cps))


def det_ediv_batch(series):
    """Batches all series into a single Rscript call (ecp::e.divisive)."""
    d = tempfile.mkdtemp()
    for i, X in enumerate(series):
        np.savetxt(os.path.join(d, f"s{i}.csv"), X, delimiter=",")
    rs = f'''suppressMessages(library(ecp))
    for (i in 0:{len(series)-1}) {{
      X <- as.matrix(read.csv(file.path("{d}", paste0("s", i, ".csv")), header=FALSE))
      res <- e.divisive(X, sig.lvl=0.05, R=199, min.size=30, alpha=1)
      est <- res$estimates; est <- est[est > 1 & est < nrow(X)]
      cat(i, ":", paste(est, collapse=" "), "\\n") }}'''
    out = subprocess.run(["Rscript", "-e", rs], capture_output=True, text=True)
    r = {}
    for line in out.stdout.strip().splitlines():
        if ":" in line:
            a, b = line.split(":", 1)
            r[int(a)] = [int(float(v)) for v in b.split()] if b.strip() else []
    return r


def match(true, pred, tol=TOL):
    used, tp, errs = set(), 0, []
    for t in true:
        best = None
        for j, p in enumerate(pred):
            if j in used or abs(p - t) > tol:
                continue
            if best is None or abs(p - t) < abs(pred[best] - t):
                best = j
        if best is not None:
            used.add(best); tp += 1; errs.append(abs(pred[best] - t))
    fp, fn = len(pred) - tp, len(true) - tp
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not true else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mae = float(np.mean(errs)) if errs else np.nan
    return prec, rec, f1, mae


def covering(true_cps, pred_cps, n=N):
    def segs(cps):
        b = [0] + list(cps) + [n]
        return [set(range(b[i], b[i + 1])) for i in range(len(b) - 1)]
    A, Ap = segs(true_cps), segs(pred_cps)
    return sum(len(a) * max(len(a & x) / len(a | x) for x in Ap) for a in A) / n


def metrics_row(family, dim, k, method, trial, true, pred, dt):
    prec, rec, f1, mae = match(true, pred)
    return {"family": family, "dim": dim, "k": k, "method": method, "trial": trial,
            "k_hat": len(pred), "exact": int(len(pred) == k), "abs_kerr": abs(len(pred) - k),
            "precision": prec, "recall": rec, "f1": f1, "mae": mae,
            "covering": covering(true, pred), "false_alarm": int(k == 0 and len(pred) > 0),
            "time_s": dt}


def run(trials):
    rows, done = [], set()
    if os.path.exists(CKPT):
        prev = pd.read_csv(CKPT); rows = prev.to_dict("records")
        done = set(zip(prev.family, prev.dim, prev.k, prev.method))
    for family in FAMILIES:
        for dim in DIMS:
            for k in KVALS:
                data = [make(family, k, dim, seed=12000 + 1000 * k + t) for t in range(trials)]
                series = [d[0] for d in data]; trues = [d[1] for d in data]
                for method in METHODS:
                    if (family, dim, k, method) in done:
                        continue
                    t0 = time.time()
                    if method == "edivisive":
                        preds = det_ediv_batch(series)
                        dt_each = (time.time() - t0) / trials
                        for t in range(trials):
                            rows.append(metrics_row(family, dim, k, method, t, trues[t], preds.get(t, []), dt_each))
                    else:
                        for t in range(trials):
                            tt = time.time()
                            if method == "ecf":
                                pred = det_ecf(series[t], dim, t)
                            else:
                                pred = det_midast(series[t], "ks" if method == "midast_ks" else "cvm")
                            rows.append(metrics_row(family, dim, k, method, t, trues[t], pred, time.time() - tt))
                    pd.DataFrame(rows).to_csv(CKPT, index=False)
                    print(f"done {family} d={dim} k={k} {method} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL CELLS DONE", flush=True)


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(CKPT)
    NAMES = {"ecf": "ECF", "midast_ks": "MIDAST[KS]", "midast_cvm": "MIDAST[CvM]", "edivisive": "e-Divisive"}

    agg = df.groupby(["family", "dim", "k", "method"]).agg(
        exact_pct=("exact", lambda x: 100 * x.mean()), mean_khat=("k_hat", "mean"),
        f1=("f1", "mean"), covering=("covering", "mean"), mae=("mae", "mean"),
        time_s=("time_s", "mean")).reset_index()
    agg.to_csv(os.path.join(OUTDIR, "summary_percell.csv"), index=False)

    far = df[df.k == 0].groupby(["family", "dim", "method"])["false_alarm"].mean().mul(100).reset_index()
    far.to_csv(os.path.join(OUTDIR, "false_alarm.csv"), index=False)

    head = df.groupby(["dim", "k", "method"]).agg(
        exact_pct=("exact", lambda x: 100 * x.mean()), f1=("f1", "mean"),
        covering=("covering", "mean"), mae=("mae", "mean")).reset_index()
    head.to_csv(os.path.join(OUTDIR, "headline.csv"), index=False)

    for metric, label in [("f1", "F1 @tol=25"), ("covering", "Covering metric"), ("exact", "Exact-k %")]:
        fig, axes = plt.subplots(len(DIMS), len(FAMILIES), figsize=(4.2 * len(FAMILIES), 3.6 * len(DIMS)), squeeze=False)
        for i, dim in enumerate(DIMS):
            for j, fam in enumerate(FAMILIES):
                ax = axes[i][j]
                for m in METHODS:
                    s = df[(df.family == fam) & (df.dim == dim) & (df.method == m)]
                    ys = [(100 * s[s.k == k]["exact"].mean() if metric == "exact" else s[s.k == k][metric].mean()) for k in KVALS]
                    ax.plot(KVALS, ys, marker="o", label=NAMES[m])
                ax.set_title(f"{fam}, d={dim}", fontsize=9)
                ax.set_xlabel("true k"); ax.set_xticks(KVALS)
                ax.set_ylabel(label); ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        plt.suptitle(f"Unknown-k benchmark: {label} vs k", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"fig_{metric}.png"), dpi=140)
        plt.close()

    print("\n=== HEADLINE (avg over families) ===")
    for metric in ["f1", "covering"]:
        print(f"\n[{metric}]")
        for dim in DIMS:
            piv = head[head.dim == dim].pivot(index="k", columns="method", values=metric)[METHODS]
            print(f" dim={dim}\n{piv.round(3).to_string()}")
    print("\n=== FALSE-ALARM rate % at k=0 ===")
    print(far.pivot_table(index=["family", "dim"], columns="method", values="false_alarm")[METHODS].round(0).to_string())
    print(f"\nSaved tables and figures to {OUTDIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.trials)
        summarize()
