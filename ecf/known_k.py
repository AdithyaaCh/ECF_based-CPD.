"""ECF change-point detector for a known single change (k=1).

Projects each window onto M random Fourier frequencies at several scales,
averages the cosine features into a fingerprint of the window's empirical
characteristic function, and scores each candidate location by the cosine
dissimilarity between its past and future window fingerprints. The reported
change-point is the most prominent peak of the resulting score curve.

Cosine-only by default: the sine (imaginary-CF) component carries no signal
for symmetric distributions and was found empirically to only add variance;
see the ablation study referenced in the paper.
"""
import numpy as np
from scipy.signal import find_peaks

M_FREQ = 256
SCALES = (0.5, 1.0, 2.0)
SEED = 0


class ECF:
    def __init__(self, d, M=M_FREQ, scales=SCALES, seed=SEED,
                 window=150, gap=10, scan_step=5, smooth=5, feature="cos"):
        rng = np.random.default_rng(seed)
        self.U = (rng.standard_normal((M, d)) / np.sqrt(d)).astype(np.float32)
        self.scales = scales
        self.window = window
        self.gap = gap
        self.scan_step = scan_step
        self.smooth = smooth
        self.feature = feature  # "cos" (default) or "cossin"

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        return ((X - med) / scale).astype(np.float32)

    def score_series(self, X, L=None, gap=None, step=None, smooth=None):
        L = self.window if L is None else L
        gap = self.gap if gap is None else gap
        step = self.scan_step if step is None else step
        smooth = self.smooth if smooth is None else smooth

        Xs = self._robust_standardize(X)
        n = len(Xs)
        idx = np.arange(L, n - L - gap, step)
        T = len(idx)
        if T == 0:
            return idx, np.empty(0)

        past_wins = np.stack([Xs[t - L:t] for t in idx])
        future_wins = np.stack([Xs[t + gap:t + L + gap] for t in idx])

        feats_p, feats_f = [], []
        use_sin = (self.feature == "cossin")
        for s in self.scales:
            Us = self.U * np.float32(s)
            Sp = past_wins @ Us.T
            feats_p.append(np.cos(Sp).mean(axis=1))
            if use_sin:
                feats_p.append(np.sin(Sp).mean(axis=1))
            Sf = future_wins @ Us.T
            feats_f.append(np.cos(Sf).mean(axis=1))
            if use_sin:
                feats_f.append(np.sin(Sf).mean(axis=1))

        zp = np.concatenate(feats_p, axis=1)
        zf = np.concatenate(feats_f, axis=1)
        zp /= np.linalg.norm(zp, axis=1, keepdims=True) + 1e-12
        zf /= np.linalg.norm(zf, axis=1, keepdims=True) + 1e-12
        scores = 1.0 - np.einsum("ti,ti->t", zp, zf)

        if smooth > 1 and len(scores) >= smooth:
            scores = np.convolve(scores, np.ones(smooth) / smooth, mode="same")
        return idx, scores

    def detect_known_k(self, X, n_cps=1):
        idx, scores = self.score_series(X)
        if len(scores) == 0:
            return []
        peaks, props = find_peaks(scores, prominence=1e-9,
                                  distance=max(1, self.window // self.scan_step))
        if len(peaks) == 0:
            return [int(idx[int(np.argmax(scores))])]
        order = np.argsort(props["prominences"])[::-1]
        chosen = peaks[order[:n_cps]]
        return [int(idx[p]) for p in chosen]


def mae_known_k(true_cps, pred_cps):
    if len(pred_cps) == 0:
        return np.nan
    if len(true_cps) == 1:
        return float(abs(pred_cps[0] - true_cps[0]))
    errs = [min(abs(p - t) for t in true_cps) for p in pred_cps]
    return float(np.mean(errs))
