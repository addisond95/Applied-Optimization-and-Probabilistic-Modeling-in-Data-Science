"""
scripts/run_bayesian_network.py
================================
Build the medical Bayesian Network, run exact and approximate inference for
four clinical queries, then benchmark both inference algorithms as the
network scales up.

Run from the repository root:
    python scripts/run_bayesian_network.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bayesian_network.inference import approximate_inference, exact_inference
from src.bayesian_network.model import build_medical_model
from src.bayesian_network.runtime_analysis import plot_runtime_comparison


QUERIES = [
    {
        "variables": ["COVID-19"],
        "evidence": {"Fever": 1, "Cough": 1},
        "label": "P(COVID-19 | Fever=T, Cough=T)",
    },
    {
        "variables": ["Flu"],
        "evidence": {"Fever": 1, "Cough": 0},
        "label": "P(Flu | Fever=T, Cough=F)",
    },
    {
        "variables": ["Treatment"],
        "evidence": {"Cough": 1},
        "label": "P(Treatment | Cough=T)",
    },
    {
        "variables": ["Recovery"],
        "evidence": {"Fever": 1, "Treatment": 1},
        "label": "P(Recovery | Fever=T, Treatment=T)",
    },
]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Build and validate the network
    # ------------------------------------------------------------------
    print("Building medical Bayesian Network …")
    model = build_medical_model()
    print("  Network is valid.\n")

    # ------------------------------------------------------------------
    # 2. Inference queries
    # ------------------------------------------------------------------
    for q in QUERIES:
        print(f"Query: {q['label']}")

        print("  Exact (Variable Elimination):")
        result_exact = exact_inference(model, q["variables"], q["evidence"])
        print(f"  {result_exact}\n")

        print("  Approximate (Likelihood Weighted Sampling, n=5000):")
        result_approx = approximate_inference(
            model, q["variables"], q["evidence"], size=5000
        )
        for var, dist in result_approx.items():
            for state, prob in dist.items():
                label = "True" if state == 1 else "False"
                print(f"    P({var}={label}) ≈ {prob:.4f}")
        print()

    # ------------------------------------------------------------------
    # 3. Runtime scaling benchmark
    # ------------------------------------------------------------------
    print("Benchmarking inference runtime as network size grows …")
    sizes, ve_times, lw_times = plot_runtime_comparison(
        network_sizes=[2, 4, 6, 8, 10, 12]
    )

    print("\n--- Runtime Summary ---")
    print(f"{'Size':>6}  {'VE (s)':>10}  {'LWS (s)':>10}")
    for s, v, l in zip(sizes, ve_times, lw_times):
        v_str = f"{v:.4f}" if v is not None else "  failed"
        l_str = f"{l:.4f}" if l is not None else "  failed"
        print(f"{s:>6}  {v_str:>10}  {l_str:>10}")


if __name__ == "__main__":
    main()
