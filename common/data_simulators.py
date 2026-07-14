from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from vendored_midast.stblrnd import stblrnd


# ---------------------------------------------------------------------------
# Student-t (paper Eq. 19)
# ---------------------------------------------------------------------------
@dataclass
class StudentTConfig:
    n_samples: int = 1000
    n_star: int = 500
    p: int = 2
    rho_pre: float = 0.9
    rho_post: float = 0.0
    nu_pre: float = 3.0
    nu_post: float = 3.0


def _safe_corr(p: int, rho: float) -> np.ndarray:
    """Build a p x p correlation matrix with off-diagonal `rho`, ridge-corrected
    if not strictly positive-definite (necessary for high d with negative rho)."""
    sigma = np.full((p, p), rho, dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)
    eig_min = float(np.linalg.eigvalsh(sigma).min().real)
    if eig_min < 1e-6:
        sigma = sigma + (abs(eig_min) + 1e-5) * np.eye(p)
    return sigma


def generate_student_t_segment(nu: float, rho: float, n: int, p: int) -> np.ndarray:
    sigma = _safe_corr(p, rho)
    G = np.random.multivariate_normal(np.zeros(p), sigma, size=n, check_valid="ignore")
    chi2 = np.random.chisquare(df=nu, size=(n, 1))
    return np.sqrt(nu / chi2) * G


def sample_student_t_series(cfg: StudentTConfig) -> Tuple[np.ndarray, int]:
    seg1 = generate_student_t_segment(cfg.nu_pre, cfg.rho_pre, cfg.n_star, cfg.p)
    seg2 = generate_student_t_segment(cfg.nu_post, cfg.rho_post, cfg.n_samples - cfg.n_star, cfg.p)
    return np.vstack([seg1, seg2]), cfg.n_star


# ---------------------------------------------------------------------------
# Sub-Gaussian alpha-stable (paper Eq. 14)
# ---------------------------------------------------------------------------
@dataclass
class SubGaussianConfig:
    n_samples: int = 1000
    n_star: int = 500
    p: int = 2
    rho_pre: float = 0.9
    rho_post: float = 0.0
    alpha_pre: float = 1.8
    alpha_post: float = 1.8


def generate_subgaussian_segment(alpha: float, rho: float, n: int, p: int) -> np.ndarray:
    sigma = _safe_corr(p, rho)
    G = np.random.multivariate_normal(np.zeros(p), sigma, size=n, check_valid="ignore")
    gamma_val = (np.cos(np.pi * alpha / 4.0)) ** (2.0 / alpha)
    A = stblrnd(alpha=alpha / 2.0, beta=1.0, gamma=gamma_val, delta=0.0, size=(n, 1))
    return np.sqrt(np.abs(A)) * G


def sample_subgaussian_series(cfg: SubGaussianConfig) -> Tuple[np.ndarray, int]:
    seg1 = generate_subgaussian_segment(cfg.alpha_pre, cfg.rho_pre, cfg.n_star, cfg.p)
    seg2 = generate_subgaussian_segment(cfg.alpha_post, cfg.rho_post, cfg.n_samples - cfg.n_star, cfg.p)
    return np.vstack([seg1, seg2]), cfg.n_star



def sample_multi_cp_series(
    dist: str,
    n_samples: int,
    cp_positions: list[int],
    p: int,
    regime_params: list[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a series with multiple change points.

    `regime_params[i]` configures segment i. Length(regime_params) == len(cp_positions)+1.
    Each dict must hold rho/(nu|alpha) keys depending on `dist`.
    """
    boundaries = [0] + sorted(cp_positions) + [n_samples]
    assert len(regime_params) == len(boundaries) - 1, "regime count mismatch"
    segments = []
    for i, params in enumerate(regime_params):
        n_i = boundaries[i + 1] - boundaries[i]
        if dist == "student_t":
            seg = generate_student_t_segment(params["nu"], params["rho"], n_i, p)
        elif dist == "sub_gaussian":
            seg = generate_subgaussian_segment(params["alpha"], params["rho"], n_i, p)
        else:
            raise ValueError(f"unknown dist {dist}")
        segments.append(seg)
    return np.vstack(segments), np.array(sorted(cp_positions), dtype=int)
