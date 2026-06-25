# Applied Optimization and Probabilistic Modeling in Data Science

**Author:** Addison DeSalvo  
**Program:** M.S. Data Science — Johns Hopkins University  
**Focus:** Swarm AI, Machine Learning, and Public Health Applications

---

## Overview

This project demonstrates core techniques in data-science optimization and probabilistic reasoning through four end-to-end implementations:

| Module | Topic |
|--------|-------|
| EDA & Preprocessing | Wine Quality Dataset — statistics, outlier removal, normalization, PCA |
| Naïve Bayes | Gaussian classifier built from scratch; raw vs. preprocessed performance |
| LP vs. PSO | Linear Programming (scipy) vs. Particle Swarm Optimization (pyswarms) |
| Bayesian Networks | pgmpy — model construction, exact/approximate inference, runtime scaling |

A legacy Jupyter Notebook (`Optimization_and_Probabilistic_Modeling_LEGACY.ipynb`) is kept in the repository root for reference.

---

## Repository Structure

```
.
├── Optimization_and_Probabilistic_Modeling_LEGACY.ipynb  # original notebook
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_eda.py                # Question 1 — EDA & Preprocessing
│   ├── run_naive_bayes.py        # Question 2 — Naïve Bayes classifier
│   ├── run_optimization.py       # Question 3 — LP vs. PSO
│   └── run_bayesian_network.py   # Question 4 — Bayesian Networks
└── src/
    ├── eda/
    │   ├── data_loader.py        # load_wine_data()
    │   ├── summary_stats.py      # compute_grouped_stats()
    │   ├── visualization.py      # scatter plots, Mahalanobis ellipses, PCA variance plot
    │   ├── outlier_detection.py  # Mahalanobis distance, remove_outliers()
    │   ├── normalization.py      # apply_minmax_scaling(), compare_before_after()
    │   └── pca_reduction.py      # apply_pca(), apply_pca_by_quality()
    ├── naive_bayes/
    │   ├── classifier.py         # NaiveBayesClassifier, categorize_quality()
    │   └── evaluation.py         # compute_accuracy(), quality_distribution()
    ├── optimization/
    │   ├── linear_programming.py # solve_lp()
    │   └── particle_swarm.py     # solve_pso()
    └── bayesian_network/
        ├── model.py              # build_medical_model()
        ├── inference.py          # exact_inference(), approximate_inference()
        └── runtime_analysis.py   # build_expanded_model(), timing, plot_runtime_comparison()
```

---

## Setup

```bash
# Clone the repo, then install dependencies
pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

## Running the Code

Each script is self-contained and runnable from the repository root.

```bash
# Question 1 — EDA & Preprocessing
python scripts/run_eda.py

# Question 2 — Naïve Bayes
python scripts/run_naive_bayes.py

# Question 3 — Linear Programming vs. PSO
python scripts/run_optimization.py

# Question 4 — Bayesian Networks + runtime benchmark
python scripts/run_bayesian_network.py
```

---

## Module Reference

### `src/eda`

| Function | Description |
|----------|-------------|
| `load_wine_data()` | Downloads red and white wine CSVs from UCI, combines them |
| `compute_grouped_stats(df)` | Min, max, mean, trimmed mean, std, skewness, kurtosis grouped by quality |
| `plot_scatter(df, x, y)` | Simple scatter plot |
| `plot_pairwise_ellipses(df, pairs)` | Mahalanobis 95 % confidence ellipses for feature pairs |
| `plot_cumulative_variance(ev)` | Cumulative PCA explained-variance plot |
| `compute_mahalanobis(df)` | Per-row Mahalanobis distance (all numeric features) |
| `remove_outliers(df, confidence)` | Drop rows beyond the chi-squared threshold (default 99.7 %) |
| `apply_minmax_scaling(df)` | Min-Max scale all numeric columns to [0, 1] |
| `compare_before_after(...)` | Side-by-side stats table pre/post scaling |
| `apply_pca(df, cols)` | Full-dataset PCA; returns fitted object, variance ratios, table |
| `apply_pca_by_quality(df, cols)` | Separate PCA per quality level |

### `src/naive_bayes`

| Symbol | Description |
|--------|-------------|
| `categorize_quality(q)` | Maps integer quality score → Low / Average / High |
| `NaiveBayesClassifier` | `.fit(df, features)` / `.predict(X)` — no sklearn used |
| `compute_accuracy(actual, predicted)` | Fraction of correct predictions |
| `quality_distribution(df)` | Frequency + percentage table for quality categories |

### `src/optimization`

| Function | Description |
|----------|-------------|
| `solve_lp()` | Exact LP solution via `scipy.optimize.linprog` (HiGHS method) |
| `solve_pso(n_particles, iters, options)` | PSO approximation via `pyswarms`, with constraint penalty |

**Problem:** min −4x₁ − 3x₂  subject to  x₁ + 2x₂ ≤ 8,  3x₁ + x₂ ≤ 9,  x₁, x₂ ≥ 0  
**LP result:** x* = (2, 3), f(x*) = −17.0  
**PSO result:** x* ≈ (2, 3), f(x*) ≈ −17.0 (within 10⁻⁵)

### `src/bayesian_network`

| Function | Description |
|----------|-------------|
| `build_medical_model()` | Six-node BN: Flu, COVID-19, Fever, Cough, Treatment, Recovery |
| `exact_inference(model, vars, evidence)` | Variable Elimination via pgmpy |
| `approximate_inference(model, vars, evidence, size)` | Likelihood-weighted sampling |
| `build_expanded_model(num_diseases, num_symptoms)` | Scalable BN for runtime benchmarks |
| `plot_runtime_comparison(network_sizes)` | Side-by-side VE vs. LWS runtime plots |

---

## Key Results

### EDA
- Alcohol content and volatile acidity are the strongest quality discriminators.
- Mahalanobis outlier removal at 99.7 % confidence eliminated ~3.4 % of rows.
- 7–8 principal components explain ≥ 90 % of variance across the full dataset.

### Naïve Bayes
- Accuracy on raw data: **~49.9 %** (naive baseline ≈ 43.7 %)
- Accuracy on preprocessed data: **~51.4 %** (+1.5 pp)
- Runtime complexity: **O(nd)** where n = samples, d = features

### LP vs. PSO

| Method | x1 | x2 | f(x*) | Notes |
|--------|----|----|-------|-------|
| LP (exact) | 2.0000 | 3.0000 | −17.0000 | Deterministic, milliseconds |
| PSO (approx) | ≈ 2.0 | ≈ 3.0 | ≈ −17.0 | Stochastic, ~1–2 s |

### Bayesian Network
- Exact inference: P(COVID-19=True | Fever=T, Cough=T) ≈ 38.6 %
- Variable Elimination scales exponentially with treewidth; Likelihood Weighted Sampling grows linearly with iterations and outperforms VE on large networks.

---

## Potential Extensions
- MCMC (Gibbs Sampling) for Bayesian inference
- Hybrid LP + PSO optimizer (LP initialization, PSO refinement)
- High-dimensional PSO benchmarks on multimodal functions
- Clinical decision support integration

---

## Technical Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `scikit-learn` · `Matplotlib` · `pyswarms` · `pgmpy`
