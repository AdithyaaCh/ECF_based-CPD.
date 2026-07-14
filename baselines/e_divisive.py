import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import rpy2.robjects as ro
from vendored_midast.cp_detection_R import ecp, _conv

SIG_LVL = 0.05
R_PERM = 199
MIN_SIZE = 10
ALPHA = 1


def e_divisive(X, sig_lvl=SIG_LVL, n_perm=R_PERM, min_size=MIN_SIZE, alpha=ALPHA):
    """X: (N, d) array. Returns sorted 0-indexed change-point locations."""
    df = pd.DataFrame(X)
    with _conv.context():
        rdata = ro.conversion.get_conversion().py2rpy(df.reset_index(drop=True))
        results = ecp.e_divisive(X=rdata, sig_lvl=sig_lvl, R=n_perm,
                                  min_size=min_size, alpha=alpha)
    names = results.names() if callable(results.names) else results.names
    d = dict(zip(names, list(results)))
    estimates = sorted(int(v) for v in d["estimates"])
    n = X.shape[0]
    return [e - 1 for e in estimates if e != 1 and e != n + 1]
