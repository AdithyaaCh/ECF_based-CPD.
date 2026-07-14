# NEURAL_ECF: vendored MIDAST-1.0.0 components.
# Provides MIDAST's exact ChangeDetector (KS, KS_DKW, MMD, Kernel, Cramer, Copula),
# Olivier Laurent's multivariate KS (`ks_2samp`), Zhaozhou Li's 2D KS (`ks2d2s`),
# the alpha-stable RNG `stblrnd`, and R-backed e-Divisive / KCPA wrappers.
#
# All algorithm code is verbatim from MIDAST-1.0.0; only import bootstraps were
# adapted for Linux/Kaggle and to live inside this Python package.

from .multivariate_statistical_test_method import ChangeDetector
from .ks_2samp import ks_2samp
from .ndtest import ks2d2s
from .stblrnd import stblrnd

__all__ = ["ChangeDetector", "ks_2samp", "ks2d2s", "stblrnd"]
