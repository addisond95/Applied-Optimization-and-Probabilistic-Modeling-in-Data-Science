from typing import List

import numpy as np
import pandas as pd


def categorize_quality(q: int) -> str:
    """Map a numeric quality score to a Low / Average / High label."""
    if q <= 5:
        return "Low Quality"
    elif q == 6:
        return "Average Quality"
    return "High Quality"


def _gaussian_log_likelihood(
    x: pd.Series, mean: pd.Series, std: pd.Series
) -> pd.Series:
    eps = 1e-6
    coeff = -0.5 * np.log(2 * np.pi * (std + eps) ** 2)
    exponent = -((x - mean) ** 2) / (2 * (std + eps) ** 2)
    return coeff + exponent


class NaiveBayesClassifier:
    """
    Gaussian Naïve Bayes implemented from scratch (no sklearn).

    Usage
    -----
    clf = NaiveBayesClassifier()
    clf.fit(df_train, feature_cols, target_col="quality_category")
    predictions = clf.predict(df_test)
    """

    def __init__(self) -> None:
        self.priors: pd.Series = None
        self.means: pd.DataFrame = None
        self.stds: pd.DataFrame = None
        self.features: List[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str = "quality_category",
    ) -> "NaiveBayesClassifier":
        self.features = features
        self.priors = df[target_col].value_counts(normalize=True)
        self.means = df.groupby(target_col)[features].mean()
        self.stds = df.groupby(target_col)[features].std()
        return self

    def predict(self, X: pd.DataFrame) -> List[str]:
        predictions: List[str] = []
        for _, row in X.iterrows():
            log_posteriors = {}
            for cls in self.priors.index:
                log_prior = np.log(self.priors[cls])
                log_likelihood = _gaussian_log_likelihood(
                    row[self.features], self.means.loc[cls], self.stds.loc[cls]
                ).sum()
                log_posteriors[cls] = log_prior + log_likelihood
            predictions.append(max(log_posteriors, key=log_posteriors.get))
        return predictions
