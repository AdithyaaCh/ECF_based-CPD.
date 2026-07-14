"""ECF-Divisive: binary segmentation for an unknown number of change-points.

At each node, the best candidate split is scored by the ECF fingerprint
dissimilarity (as in known_k.py) and validated by a permutation test with
exact adaptive early stopping (Besag & Clifford, 1991). Splits are accepted
recursively until no segment yields a significant split.

Permutations are applied to a small sparse averaging matrix rather than to
the feature matrix itself (S[:, inv] @ seg == S @ seg[perm]), which is
algebraically exact and far cheaper since the averaging matrix is much
smaller than the feature matrix.
"""
import numpy as np

M_FREQ = 256
SCALES = (0.5, 1.0, 2.0)
SEED = 0

WINDOW, GAP, STEP, SMOOTH = 100, 10, 5, 5
ALPHA_SIG = 0.05
N_PERM = 150
MIN_SEG = 2 * WINDOW + GAP + STEP


class ECFCore:
    def __init__(self, d, M=M_FREQ, scales=SCALES, seed=SEED, feature="cos"):
        rng = np.random.default_rng(seed)
        self.U = (rng.standard_normal((M, d)) / np.sqrt(d)).astype(np.float32)
        self.scales = scales
        self.feature = feature  # "cos" (default) or "cossin"

    @staticmethod
    def robust_standardize(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        return ((X - med) / scale).astype(np.float32)

    def feature_matrix(self, Xs):
        feats = []
        for s in self.scales:
            proj = Xs @ (self.U * np.float32(s)).T
            feats.append(np.cos(proj))
            if self.feature == "cossin":
                feats.append(np.sin(proj))
        return np.hstack(feats).astype(np.float32)

    @staticmethod
    def _build_S(m, window, gap, step):
        idx = np.arange(window, m - window - gap, step)
        T = len(idx)
        Sb = np.zeros((T, m), dtype=np.float32)
        Sa = np.zeros((T, m), dtype=np.float32)
        for i, t in enumerate(idx):
            Sb[i, t - window:t] = 1.0 / window
            Sa[i, t + gap:t + window + gap] = 1.0 / window
        return idx, np.vstack([Sb, Sa]), T

    @staticmethod
    def _curve_from_S(S_stack, T, seg, smooth):
        BA = S_stack @ seg
        b, a = BA[:T], BA[T:]
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        c = 1.0 - np.sum(b * a, axis=1)
        if smooth > 1 and len(c) >= smooth:
            c = np.convolve(c, np.ones(smooth) / smooth, mode="same")
        return c


# alias, no functional difference -- kept for readability at call sites
ECFDivisive = ECFCore


def ecf_divisive(feat, rng, window=WINDOW, gap=GAP, step=STEP, smooth=SMOOTH,
                 alpha=ALPHA_SIG, n_perm=N_PERM, min_seg=MIN_SEG):
    found = []
    S_cache = {}
    stop_thresh = int(np.floor(alpha * (n_perm + 1))) - 1

    def get_S(m):
        if m not in S_cache:
            S_cache[m] = ECFCore._build_S(m, window, gap, step)
        return S_cache[m]

    def recurse(lo, hi):
        m = hi - lo
        if m < min_seg:
            return
        idx, S_stack, T = get_S(m)
        if T == 0:
            return
        seg = feat[lo:hi]
        obs_curve = ECFCore._curve_from_S(S_stack, T, seg, smooth)
        best = int(np.argmax(obs_curve))
        obs = float(obs_curve[best])
        cp = int(idx[best] + lo)

        ge = 0
        used = 0
        for _ in range(n_perm):
            perm = rng.permutation(m)
            inv = np.argsort(perm)
            Sp = S_stack[:, inv]
            c = ECFCore._curve_from_S(Sp, T, seg, smooth)
            used += 1
            if c.max() >= obs:
                ge += 1
                if ge > stop_thresh:
                    break
        pval = (ge + 1) / (used + 1)
        if pval > alpha:
            return
        found.append(cp)
        recurse(lo, cp)
        recurse(cp, hi)

    recurse(0, feat.shape[0])
    return sorted(found)


def match_cps(true_cps, pred_cps, tol=40):
    """Greedy nearest match within tol. Returns (tp, fp, fn, matched_errs)."""
    true_left = list(true_cps)
    errs = []
    tp = 0
    for p in sorted(pred_cps):
        if not true_left:
            break
        dists = [abs(p - t) for t in true_left]
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            tp += 1
            errs.append(dists[j])
            true_left.pop(j)
    fp = len(pred_cps) - tp
    fn = len(true_cps) - tp
    return tp, fp, fn, errs


def mae_exact_k(true_cps, pred_cps):
    """Defined only when the predicted and true counts match."""
    if len(pred_cps) != len(true_cps) or len(true_cps) == 0:
        return float("nan")
    return float(np.mean([abs(p - t) for p, t in zip(sorted(pred_cps), sorted(true_cps))]))
