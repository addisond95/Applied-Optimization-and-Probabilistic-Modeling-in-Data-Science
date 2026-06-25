import pandas as pd


def compute_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Return the fraction of correctly classified samples."""
    return round((actual == predicted).sum() / len(actual), 4)


def quality_distribution(df: pd.DataFrame, col: str = "quality_category") -> pd.DataFrame:
    """Return a frequency + percentage table for the quality category column."""
    freq = df[col].value_counts().reset_index()
    freq.columns = ["Quality Category", "Frequency"]
    pct = df[col].value_counts(normalize=True).reset_index()
    pct.columns = ["Quality Category", "Percentage"]
    pct["Percentage"] = (pct["Percentage"] * 100).round(2)
    return freq.merge(pct, on="Quality Category")
