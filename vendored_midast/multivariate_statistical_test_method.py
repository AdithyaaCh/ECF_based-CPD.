from dotenv import load_dotenv

load_dotenv()

import os
from typing import Union

import numpy as np
import pandas as pd
from tqdm import trange

USE_R_TEST = bool(os.getenv("USE_R_TESTS"))
if USE_R_TEST:
    from .multivariate_tests_from_R import CopulaTest, CramerTest, KernelDensitiesTest, MMDTest

# `topotest` is an optional R/topology dependency. Lazy-import it inside
# the only branch that uses it, so dropping it does not break MIDAST[KS|MMD|...].
def _lazy_TopoTestTwosample(*args, **kwargs):
    from topotest import TopoTestTwosample  # imported only on demand
    return TopoTestTwosample(*args, **kwargs)

from .ks_2samp import ks_2samp
from .ndtest import ks2d2s


class ChangeDetector:
    def __init__(
        self,
        test_name: str = "KSTest",
        bn: int = 200,
    ) -> None:
        """
        Initialize the ChangeDetector class.

        Args:
            test_name (str): Name of the statistical test to use. Default is "KSTest".
            bn (int): Number of bootstrap samples. Default is 200.
        """
        # self.alpha = alpha
        self.boot_num = bn
        self.window_size = None
        self.test_name = test_name

    def test_in_window(
        self,
        df: np.ndarray,
        window_size: int,
        shift: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform statistical tests on sliding windows of the input data.

        Args:
            df (np.ndarray): Input data as a NumPy array.
            window_size (int): Size of the sliding window.
            shift (int): Step size for the sliding window. Default is 10.
            **kwargs: Additional arguments for specific tests.

        Returns:
            pd.DataFrame: Results containing statistics and p-values for each window.
        """
        results_statictic = {}
        results_pvalue = {}

        n_rows, dimension = df.shape

        for ind in trange(0, n_rows - 2 * window_size, shift):
            values1 = df[ind : ind + window_size, :]
            values2 = df[ind + window_size : ind + 2 * window_size, :]

            if (self.test_name == "KSTest") or (self.test_name == "KSTest_DKW") :
                if (dimension == 2) and (self.test_name == "KSTest"):
                    x1, y1 = values1.T
                    x2, y2 = values2.T
                    result = ks2d2s(x1, y1, x2, y2, extra=True)
                else:
                    stat, _, _, pvalue = ks_2samp(x_val=values1, y_val=values2, alpha=0.05)  # alpha=self.alpha)
                    result = [pvalue, stat]
            elif self.test_name == "TopologyTest":
                res = _lazy_TopoTestTwosample(X1=values1, X2=values2)
                result = [res.pvalue, res.statistic]
            elif self.test_name == "KernelDensitiesTest":
                if USE_R_TEST:
                    test_instance = KernelDensitiesTest(
                        df1=pd.DataFrame(values1),
                        df2=pd.DataFrame(values2),
                    )
                    result = test_instance.conduct_test(boot_num=self.boot_num)
                else:
                    raise ValueError(
                        "USE_R_TEST set as False. Please download and configure R-dependencies firstly, change R_PATH in .env set USE_R_TEST as True - if you want to use KernelDensitiesTest, MMDTest or Cramer-von-Mises test."
                    )
            elif self.test_name == "MMDTest":
                if USE_R_TEST:
                    test_instance = MMDTest(
                        df1=pd.DataFrame(values1),
                        df2=pd.DataFrame(values2),
                    )
                    result = test_instance.conduct_test()
                else:
                    raise ValueError(
                        "USE_R_TEST set as False. Please download and configure R-dependencies firstly, change R_PATH in .env set USE_R_TEST as True - if you want to use KernelDensitiesTest, MMDTest or Cramer-von-Mises test."
                    )
            elif self.test_name == "CramerTest":
                if USE_R_TEST:
                    test_instance = CramerTest(
                        values1=values1,
                        values2=values2,
                    )
                    result = test_instance.conduct_test(nboot=self.boot_num, **kwargs)
                else:
                    raise ValueError(
                        "USE_R_TEST set as False. Please download and configure R-dependencies firstly, change R_PATH in .env set USE_R_TEST as True - if you want to use KernelDensitiesTest, MMDTest or Cramer-von-Mises test."
                    )
            elif self.test_name == "CopulaTest":
                if USE_R_TEST:
                    test_instance = CopulaTest(
                        values1=values1,
                        values2=values2,
                    )
                    result = test_instance.conduct_test(boot_num=self.boot_num)
                else:
                    raise ValueError(
                        "USE_R_TEST set as False. Please download and configure R-dependencies firstly, change R_PATH in .env set USE_R_TEST as True - if you want to use KernelDensitiesTest, MMDTest or Cramer-von-Mises test."
                    )
            else:
                raise (NotImplementedError)

            results_statictic[ind + window_size] = result[1]
            results_pvalue[ind + window_size] = result[0]

        results_df = pd.DataFrame()
        results_df["id"] = results_statictic.keys()
        results_df["window1_start"] = [el - window_size for el in results_statictic]
        results_df["window2_end"] = [el + window_size for el in results_statictic]
        results_df["statistic"] = results_statictic.values()
        results_df["pvalue"] = results_pvalue.values()

        return results_df

    def fit(
        self,
        df: pd.DataFrame,
        window_size: int,
        shift: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fit the change detection model on the input data.

        Args:
            df (pd.DataFrame): Input data as a Pandas DataFrame.
            window_size (int): Size of the sliding window.
            shift (int): Step size for the sliding window. Default is 10.
            **kwargs: Additional arguments for specific tests.

        Returns:
            pd.DataFrame: Results containing statistics and p-values for each window.
        """
        results_df = self.test_in_window(df=df, window_size=window_size, shift=shift, **kwargs)

        self.window_size = window_size

        self.results_df = results_df
        return results_df

    def analyze_results(
        self,
        results_df: pd.DataFrame,
        output_type: str = "np.array",
        alpha: float = 0.05,
        shift_group: int | None = None,
        max_no_changes: int | None = None,
        max_pvalues_for_grouping: int | None = None,
        based_on: str = "statistic",
    ) -> Union[list, pd.DataFrame, None]:
        """
        Analyze the results of the statistical tests to identify change points.

        Args:
            results_df (pd.DataFrame): DataFrame containing test results.
            output_type (str): Output format, either "np.array" or "pd.DataFrame". Default is "np.array".
            alpha (float): Significance level for p-value filtering. Default is 0.05.
            shift_group (int | None): Maximum distance between change points to group them. Default is None.
            max_no_changes (int | None): Maximum number of change points to return. Default is None.
            max_pvalues_for_grouping (int | None): Maximum number of p-values to consider for grouping. Default is None.
            based_on (str): Criterion for selecting change points, either "statistic" or "pvalue". Default is "statistic".

        Returns:
            Union[list, pd.DataFrame, None]: Change points as a list or DataFrame, or None if no change points are found.
        """
        if max_pvalues_for_grouping:
            change_points = (
                results_df[results_df.pvalue <= alpha]
                .sort_values(by="pvalue")
                .reset_index(drop=True)
                .loc[:max_pvalues_for_grouping, :]
                .sort_values(by="id")
                .reset_index(drop=True)
            )
        else:
            change_points = results_df[results_df.pvalue <= alpha]

        if not change_points.empty:
            change_points["group"] = None
            values = change_points.id.values

            if not shift_group:
                shift_group = self.window_size

            groups = {}
            id = 1
            el = [values[0]]
            for item in values[1:]:
                if item - el[-1] <= shift_group:
                    el.append(item)
                else:
                    groups[id] = el
                    el = [item]
                    id += 1
            groups[id] = el

            for key in groups:
                ids = results_df[results_df.id.isin(groups[key])].index
                results_df.loc[ids, "group"] = key

            if based_on == "pvalue":
                cp_values = results_df.groupby(by="group")["pvalue"].min().reset_index().values
                cp_id = []
                for item in cp_values:
                    group_name, pmin = item
                    cp_id.append(
                        np.round(
                            np.median(results_df[(results_df.group == group_name) & (results_df.pvalue == pmin)].index)
                        )
                    )
            elif based_on == "statistic":
                cp_values = results_df.groupby(by="group")["statistic"].max().reset_index().values
                cp_id = []
                for item in cp_values:
                    group_name, stat_max = item
                    cp_id.append(
                        np.round(
                            np.median(
                                results_df[(results_df.group == group_name) & (results_df.statistic == stat_max)].index
                            )
                        )
                    )

            else:
                raise ValueError("Parameter 'based_on' should be 'statistic' or 'pvalue'.")

            cp = results_df.loc[cp_id]

            if max_no_changes:
                if based_on == "pvalue":
                    cp = cp.sort_values(by="pvalue").head(max_no_changes)
                else:
                    cp = cp.sort_values(by="statistic", ascending=False).head(max_no_changes)

            if output_type == "pd.DataFrame":
                return cp

            elif output_type == "np.array":
                return cp.id.values
            else:
                raise (ValueError("'output_type' undefined. Please choose 'pd.DataFrame' or 'np.array'"))
        else:
            return pd.DataFrame() if output_type == "pd.DataFrame" else None
