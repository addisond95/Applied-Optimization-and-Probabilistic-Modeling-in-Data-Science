from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def apply_minmax_scaling(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Apply Min-Max normalization to all numeric columns.

    Returns
    -------
    df_scaled    : scaled copy of the DataFrame
    numeric_cols : index of the scaled column names
    """
    numeric_cols = df.select_dtypes(include="number").columns
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, numeric_cols


def compare_before_after(
    df_original: pd.DataFrame,
    df_scaled: pd.DataFrame,
    numeric_cols: pd.Index,
) -> pd.DataFrame:
    """Return a side-by-side statistics table (before vs after scaling)."""
    stats_before = df_original[numeric_cols].agg(["min", "max", "mean", "std"]).T
    stats_before.columns = ["min_before", "max_before", "mean_before", "std_before"]

    stats_after = df_scaled[numeric_cols].agg(["min", "max", "mean", "std"]).T
    stats_after.columns = ["min_after", "max_after", "mean_after", "std_after"]

    return pd.concat([stats_before, stats_after], axis=1).round(4)
