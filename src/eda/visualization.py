from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import chi2


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    color: str = "steelblue",
) -> None:
    """Render a scatter plot of two columns."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5, color=color)
    plt.title(title or f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _draw_mahalanobis_ellipse(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    confidence: float = 0.95,
) -> None:
    """Overlay a Mahalanobis confidence ellipse on an axes object."""
    X = data[[x_col, y_col]].dropna().values
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    radius = np.sqrt(chi2.ppf(confidence, df=2))

    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = mean[:, None] + radius * np.linalg.cholesky(cov) @ circle

    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10)
    ax.plot(ellipse[0], ellipse[1], color="red")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} (Mahalanobis Ellipse)")


def plot_pairwise_ellipses(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Plot Mahalanobis ellipses for a list of feature pairs."""
    if pairs is None:
        pairs = [
            ("alcohol", "quality"),
            ("volatile acidity", "citric acid"),
            ("residual sugar", "density"),
        ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (x_col, y_col) in zip(axes, pairs):
        _draw_mahalanobis_ellipse(df, x_col, y_col, ax)

    plt.tight_layout()
    plt.show()


def plot_cumulative_variance(explained_variance: np.ndarray) -> None:
    """Plot cumulative explained variance with 90 % and 95 % threshold lines."""
    cumulative = np.cumsum(explained_variance)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.title("Cumulative Explained Variance by Principal Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.90, color="red", linestyle="--", label="90% Threshold")
    plt.axhline(y=0.95, color="green", linestyle="--", label="95% Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
