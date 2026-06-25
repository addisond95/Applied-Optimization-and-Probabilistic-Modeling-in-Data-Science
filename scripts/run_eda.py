"""
scripts/run_eda.py
=================
End-to-end EDA and preprocessing pipeline on the UCI Wine Quality Dataset.

Run from the repository root:
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eda.data_loader import load_wine_data
from src.eda.normalization import apply_minmax_scaling, compare_before_after
from src.eda.outlier_detection import remove_outliers
from src.eda.pca_reduction import apply_pca, apply_pca_by_quality
from src.eda.summary_stats import compute_grouped_stats
from src.eda.visualization import (
    plot_cumulative_variance,
    plot_pairwise_ellipses,
    plot_scatter,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading Wine Quality Dataset …")
    df = load_wine_data()
    print(f"  Combined shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Summary statistics grouped by quality
    # ------------------------------------------------------------------
    print("\nComputing summary statistics …")
    stats = compute_grouped_stats(df)
    print(stats.head())

    # ------------------------------------------------------------------
    # 3. Visualisations
    # ------------------------------------------------------------------
    print("\nGenerating scatter plots …")
    plot_scatter(df, "quality", "alcohol", title="Alcohol Content vs Wine Quality")
    plot_scatter(
        df,
        "quality",
        "volatile acidity",
        title="Volatile Acidity vs Wine Quality",
        color="orange",
    )

    print("Plotting Mahalanobis confidence ellipses …")
    plot_pairwise_ellipses(df)

    # ------------------------------------------------------------------
    # 4. Outlier removal (Mahalanobis, 99.7 % threshold)
    # ------------------------------------------------------------------
    print("\nRemoving outliers …")
    df_clean, n_removed, pct = remove_outliers(df)
    print(f"  Removed {n_removed} rows ({pct:.2f} %)")
    print(f"  Cleaned shape: {df_clean.shape}")

    # ------------------------------------------------------------------
    # 5. Min-Max normalisation
    # ------------------------------------------------------------------
    print("\nApplying Min-Max normalisation …")
    df_scaled, numeric_cols = apply_minmax_scaling(df_clean)
    comparison = compare_before_after(df_clean, df_scaled, numeric_cols)
    print(comparison)

    # Sanity check: no values outside [0, 1]
    out_of_bounds = ((df_scaled[numeric_cols] < 0) | (df_scaled[numeric_cols] > 1)).any()
    print("\nAny values outside [0, 1]:", out_of_bounds.any())

    # ------------------------------------------------------------------
    # 6. PCA on full dataset
    # ------------------------------------------------------------------
    print("\nApplying PCA to full normalised dataset …")
    _, explained_variance, pca_table = apply_pca(df_scaled, numeric_cols)
    print(pca_table.to_string(index=False))
    plot_cumulative_variance(explained_variance)

    # ------------------------------------------------------------------
    # 7. PCA per quality level
    # ------------------------------------------------------------------
    print("\nApplying PCA per quality level …")
    pca_quality = apply_pca_by_quality(df_scaled, numeric_cols)
    print(pca_quality.iloc[:, :5].round(4))


if __name__ == "__main__":
    main()
