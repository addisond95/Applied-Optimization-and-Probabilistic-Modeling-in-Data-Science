import warnings

import pandas as pd
from scipy.stats import kurtosis, skew, trim_mean

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _trimmed_mean(x):
    return trim_mean(x, proportiontocut=0.05)


def compute_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute min, max, mean, trimmed mean (5%), std, skewness, and kurtosis
    for each numeric feature grouped by wine quality rating.
    """
    numeric_features = df.select_dtypes(include="number").columns
    grouped_stats = df.groupby("quality")[numeric_features].agg(
        ["min", "max", "mean", _trimmed_mean, "std", skew, kurtosis]
    )
    grouped_stats.columns = [
        "_".join(col).strip() for col in grouped_stats.columns.values
    ]
    return grouped_stats
