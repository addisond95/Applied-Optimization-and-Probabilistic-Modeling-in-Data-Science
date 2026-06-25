from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(
    df_scaled: pd.DataFrame,
    numeric_cols: pd.Index,
) -> Tuple[PCA, np.ndarray, pd.DataFrame]:
    """
    Fit PCA on the full scaled dataset.

    Returns
    -------
    pca                 : fitted PCA object
    explained_variance  : array of explained variance ratios
    variance_table      : DataFrame summarising each component
    """
    X_scaled = df_scaled[numeric_cols]
    pca = PCA()
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    variance_table = pd.DataFrame(
        {
            "Principal Component": [
                f"PC{i + 1}" for i in range(len(explained_variance))
            ],
            "Explained Variance Ratio": explained_variance,
        }
    )
    return pca, explained_variance, variance_table


def apply_pca_by_quality(
    df_scaled: pd.DataFrame,
    numeric_cols: pd.Index,
) -> pd.DataFrame:
    """
    Fit a separate PCA for each wine quality level and return a DataFrame
    of explained variance ratios (rows = quality levels, cols = PCs).
    """
    quality_levels = sorted(df_scaled["quality"].unique())
    pca_by_quality: dict = {}
    max_len = 0

    for q in quality_levels:
        group_data = df_scaled[df_scaled["quality"] == q][numeric_cols]
        pca_q = PCA()
        pca_q.fit(group_data)
        var_ratio = pca_q.explained_variance_ratio_
        pca_by_quality[q] = var_ratio
        max_len = max(max_len, len(var_ratio))

    for q in pca_by_quality:
        pad = max_len - len(pca_by_quality[q])
        if pad > 0:
            pca_by_quality[q] = np.pad(
                pca_by_quality[q], (0, pad), constant_values=np.nan
            )

    pca_quality_df = pd.DataFrame(pca_by_quality).T
    pca_quality_df.columns = [f"PC{i + 1}" for i in range(pca_quality_df.shape[1])]
    pca_quality_df.index.name = "Quality"
    return pca_quality_df
