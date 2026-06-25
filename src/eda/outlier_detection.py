from typing import Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import chi2


def compute_mahalanobis(df: pd.DataFrame) -> pd.Series:
    """Return the Mahalanobis distance for every row (numeric columns only)."""
    X = df.select_dtypes(include="number").values.astype(float)
    mean_vec = X.mean(axis=0)
    cov_matrix = np.cov(X.T)
    inv_cov = inv(cov_matrix)

    deltas = X - mean_vec
    distances = np.sqrt(np.einsum("ij,jk,ik->i", deltas, inv_cov, deltas))
    return pd.Series(distances, index=df.index)


def remove_outliers(
    df: pd.DataFrame,
    threshold_confidence: float = 0.997,
) -> Tuple[pd.DataFrame, int, float]:
    """
    Remove rows whose Mahalanobis distance exceeds the given chi-squared
    confidence threshold.

    Returns
    -------
    df_clean        : DataFrame with outliers removed
    rows_removed    : number of rows dropped
    pct_removed     : percentage of original rows dropped
    """
    n_features = df.select_dtypes(include="number").shape[1]
    distances = compute_mahalanobis(df)
    threshold = np.sqrt(chi2.ppf(threshold_confidence, df=n_features))

    outlier_mask = distances > threshold
    df_clean = df.loc[~outlier_mask].reset_index(drop=True)
    rows_removed = int(outlier_mask.sum())
    pct_removed = rows_removed / len(df) * 100
    return df_clean, rows_removed, pct_removed
