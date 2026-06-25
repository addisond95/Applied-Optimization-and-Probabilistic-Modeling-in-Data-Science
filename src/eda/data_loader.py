import pandas as pd


def load_wine_data() -> pd.DataFrame:
    """Download and combine the UCI red and white wine quality datasets."""
    url_red = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/wine-quality/winequality-red.csv"
    )
    url_white = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/wine-quality/winequality-white.csv"
    )

    df_red = pd.read_csv(url_red, sep=";")
    df_white = pd.read_csv(url_white, sep=";")
    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"

    df_wine = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    return df_wine
