"""Lomax (Pareto Type II) scale-mixture data generator.

    X = sqrt(W) * G,   G ~ N(0, Sigma(rho)),   W ~ Lomax(alpha, scale).

alpha controls tail heaviness, rho controls dependence.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LomaxConfig:
    n_samples: int = 1000
    n_star: int = 500
    p: int = 2
    rho_pre: float = 0.9
    rho_post: float = 0.0
    alpha_pre: float = 4.0
    alpha_post: float = 2.0
    scale_pre: float = 1.0
    scale_post: float = 1.0


def safe_corr(p: int, rho: float) -> np.ndarray:
    sigma = np.full((p, p), rho, dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)
    eig_min = float(np.linalg.eigvalsh(sigma).min().real)
    if eig_min < 1e-6:
        sigma = sigma + (abs(eig_min) + 1e-5) * np.eye(p)
    return sigma


def generate_lomax_segment(alpha: float, rho: float, n: int, p: int, scale: float = 1.0) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    sigma = safe_corr(p, rho)
    gaussian = np.random.multivariate_normal(np.zeros(p), sigma, size=n, check_valid="ignore")
    lomax_scale = scale * np.random.pareto(alpha, size=(n, 1))
    return np.sqrt(np.maximum(lomax_scale, 1e-12)) * gaussian


def sample_lomax_series(cfg: LomaxConfig) -> Tuple[np.ndarray, int]:
    seg1 = generate_lomax_segment(cfg.alpha_pre, cfg.rho_pre, cfg.n_star, cfg.p, cfg.scale_pre)
    seg2 = generate_lomax_segment(cfg.alpha_post, cfg.rho_post, cfg.n_samples - cfg.n_star, cfg.p, cfg.scale_post)
    return np.vstack([seg1, seg2]), cfg.n_star
