from contextlib import contextmanager
from dotenv import load_dotenv
import os

load_dotenv()

# Optional explicit R path (Windows). On Linux/Kaggle leave R_PATH unset and let
# rpy2 discover R via the default `R RHOME`.
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

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter


_CONV = default_converter + pandas2ri.converter + numpy2ri.converter

# Suppress R output context
@contextmanager
def suppress_r_output():
    from rpy2.rinterface_lib import callbacks
    original_writeconsole = callbacks.consolewrite_print
    callbacks.consolewrite_print = lambda x: None
    try:
        yield
    finally:
        callbacks.consolewrite_print = original_writeconsole

# Import R packages
utils_r = importr("utils")
base = importr("base")
utils_r.chooseCRANmirror(ind=1)

np_r = importr("np")
kernel_two_sample_test = importr("maotai")
cramer_test = importr("cramer")
copula_based_test = importr("TwoCop")


class KernelDensitiesTest:
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        self.x = df1
        self.y = df2

    def conduct_test(self, boot_num: int) -> tuple[float, float]:
        with suppress_r_output(), localconverter(_CONV):
            results = np_r.npdeneqtest(self.x, self.y, boot_num=boot_num)
        results2dict = dict(zip(results.names, list(results)))
        return results2dict["Tn.P"][0], results2dict["Tn"][0]


class MMDTest:
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        self.x = df1
        self.y = df2

    def conduct_test(self) -> tuple[float, float]:
        ro.r(
            """
            f <- function(dat1, dat2, lab) {
                dmat <- as.matrix(dist(rbind(dat1, dat2)))
                kmat <- exp(-(dmat^2)) 
                result <- mmd2test(kmat, lab)
                pvalue <- result$p.value
                statistic <- result$statistic
                return(c(pvalue, statistic))
            }
            """
        )
        lab = np.array([1] * self.x.shape[0] + [2] * self.y.shape[0])
        kernel_two_sample_test_fn = ro.globalenv["f"]
        with localconverter(_CONV):
            results = kernel_two_sample_test_fn(self.x, self.y, lab)
        return results[0], results[1]


class CramerTest:
    def __init__(self, values1: np.ndarray, values2: np.ndarray) -> None:
        self.x = values1
        self.y = values2

    def conduct_test(self, nboot: int = 1000, kernel: str = "phiLog") -> tuple[float, float]:
        ro.r(
            """
            f <- function(x, y, replicates, kernel) {
                result <- cramer.test(x, y, replicates=replicates, kernel=kernel)
                pvalue <- result$p.value
                statistic <- result$statistic
                return(c(pvalue, statistic))
            }
            """
        )
        cramer_two_sample_test_fn = ro.globalenv["f"]
        with localconverter(_CONV):
            results = cramer_two_sample_test_fn(self.x, self.y, nboot, kernel)
        return results[0], results[1]


class CopulaTest:
    def __init__(self, values1: np.ndarray, values2: np.ndarray) -> None:
        self.x = values1
        self.y = values2

    def conduct_test(self, boot_num: int) -> tuple[float, float]:
        with localconverter(_CONV):
            results = copula_based_test.TwoCop(self.x, self.y, Nsim=boot_num)
        results2dict = dict(zip(results.names, list(results)))
        return results2dict["pvalue"][0], results2dict["cvm"][0]
