"""
scripts/run_naive_bayes.py
==========================
Train and evaluate a scratch-built Gaussian Naïve Bayes classifier on both
the raw and preprocessed Wine Quality datasets.

Run from the repository root:
    python scripts/run_naive_bayes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eda.data_loader import load_wine_data
from src.eda.normalization import apply_minmax_scaling
from src.eda.outlier_detection import remove_outliers
from src.naive_bayes.classifier import NaiveBayesClassifier, categorize_quality
from src.naive_bayes.evaluation import compute_accuracy, quality_distribution


def main() -> None:
    # ------------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------------
    print("Loading data …")
    df_raw = load_wine_data()

    # Add quality category to raw data
    df_raw["quality_category"] = df_raw["quality"].apply(categorize_quality)

    # Cleaned (outliers removed)
    df_clean, n_removed, _ = remove_outliers(df_raw)
    df_clean["quality_category"] = df_clean["quality"].apply(categorize_quality)

    # Scaled version of cleaned data
    df_scaled, _ = apply_minmax_scaling(df_clean)
    df_scaled["quality_category"] = df_clean["quality_category"].values

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------
    print("\nQuality category distribution (raw data):")
    print(quality_distribution(df_raw).to_string(index=False))

    # ------------------------------------------------------------------
    # Feature columns (numeric, excluding the quality columns themselves)
    # ------------------------------------------------------------------
    features_raw = [
        c for c in df_raw.select_dtypes(include="number").columns
        if c != "quality"
    ]
    features_scaled = [
        c for c in df_scaled.select_dtypes(include="number").columns
        if c != "quality"
    ]

    # ------------------------------------------------------------------
    # Train & predict on raw data
    # ------------------------------------------------------------------
    print("\nTraining Naïve Bayes on raw data …")
    clf_raw = NaiveBayesClassifier()
    clf_raw.fit(df_raw, features_raw)
    df_raw["predicted_category"] = clf_raw.predict(df_raw)
    acc_raw = compute_accuracy(df_raw["quality_category"], df_raw["predicted_category"])
    print(f"  Accuracy (raw):        {acc_raw}")

    # ------------------------------------------------------------------
    # Train & predict on preprocessed (scaled) data
    # ------------------------------------------------------------------
    print("\nTraining Naïve Bayes on preprocessed data …")
    clf_scaled = NaiveBayesClassifier()
    clf_scaled.fit(df_scaled, features_scaled)
    df_scaled["predicted_category"] = clf_scaled.predict(df_scaled)
    acc_scaled = compute_accuracy(
        df_scaled["quality_category"], df_scaled["predicted_category"]
    )
    print(f"  Accuracy (preprocessed): {acc_scaled}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Accuracy Comparison ---")
    print(f"  Raw data:        {acc_raw}")
    print(f"  Preprocessed:    {acc_scaled}")
    print(f"  Improvement:     {round(acc_scaled - acc_raw, 4)}")


if __name__ == "__main__":
    main()
