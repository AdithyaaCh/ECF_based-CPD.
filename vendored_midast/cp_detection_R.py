
from dotenv import load_dotenv
load_dotenv()

import os

# Optional explicit R path (Windows). On Linux/Kaggle leave R_PATH unset and let
# rpy2 discover R via the default `R RHOME` lookup.
path_to_R = os.getenv("R_PATH")
if path_to_R:
    os.environ["R_HOME"] = path_to_R
    path = os.path.join(path_to_R, "bin/x64/")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except (FileNotFoundError, OSError):
            pass
    else:
        os.environ["PATH"] = f"{path}:" + os.environ.get("PATH", "")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.packages import importr

# rpy2 >= 3.5 deprecated pandas2ri.activate(); use an explicit conversion
# context instead (functionally equivalent for this module's needs).
_conv = default_converter + pandas2ri.converter

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

utils_r = importr("utils")
base = importr("base")

# `chooseCRANmirror(ind=1)` is harmless if a mirror is already set, so leave it.
try:
    utils_r.chooseCRANmirror(ind=1)
except Exception:
    pass

ecp = importr("ecp")
import numpy as np


class ChangePointDetectorECP:
    """
    A class for detecting change points in time series data using the 'ecp' R package.

    Attributes:
        data (ro.vectors.DataFrame): The input data converted to an R DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the ChangePointDetectorECP with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input data as a pandas DataFrame.
        """
        with _conv.context():
            self.data: ro.vectors.DataFrame = ro.conversion.get_conversion().py2rpy(df.reset_index(drop=True))

    def kcp(self, L: int, C: float) -> tuple[float, float]:
        """
        Performs Kernel Change Point Analysis (KCPA).

        Args:
            L (int): The minimum segment length.
            C (float): The penalty parameter.

        Returns:
            tuple[float, float]: Results of the KCPA analysis.
        """
        results = ecp.kcpa(X=np.array(self.data).T, L=L, C=C)
        return results

    def edivisive(
        self,
        sig_lvl: float = 0.05,
        R: int = 199,
        min_size: int = 10,
        alpha: float = 1,
        n_bkps: int | None = None,
    ) -> dict:
        """
        Performs e-divisive change point detection.

        Args:
            sig_lvl (float): Significance level for hypothesis testing. Default is 0.05.
            R (int): Number of permutations. Default is 199.
            min_size (int): Minimum segment size. Default is 10.
            alpha (float): Weighting parameter. Default is 1.
            n_bkps (int | None): Number of breakpoints to detect. Default is None.

        Returns:
            dict: A dictionary containing the results of the e-divisive analysis.
        """
        if n_bkps:
            results = ecp.e_divisive(
                X=self.data,
                sig_lvl=sig_lvl,
                R=R,
                min_size=min_size,
                alpha=alpha,
                k=n_bkps,
            )
        else:
            results = ecp.e_divisive(X=self.data, sig_lvl=sig_lvl, R=R, min_size=min_size, alpha=alpha)
        results2dict = dict(zip(results.names, list(results)))
        return results2dict["considered.last"]
