import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def build_expanded_model(num_diseases: int = 2, num_symptoms: int = 2) -> BayesianNetwork:
    """
    Construct a scalable Bayesian Network with `num_diseases` disease nodes
    and `num_symptoms` symptom nodes, plus a single Treatment and Recovery node.

    All diseases influence all symptoms.  The first disease drives Treatment.
    All diseases and Treatment jointly affect Recovery.
    """
    disease_nodes = [f"Disease_{i}" for i in range(num_diseases)]
    symptom_nodes = [f"Symptom_{i}" for i in range(num_symptoms)]

    edges = []
    for d in disease_nodes:
        for s in symptom_nodes:
            edges.append((d, s))
    edges.append((disease_nodes[0], "Treatment"))
    for d in disease_nodes:
        edges.append((d, "Recovery"))
    edges.append(("Treatment", "Recovery"))

    model = BayesianNetwork(edges)

    # Disease priors
    for d in disease_nodes:
        model.add_cpds(
            TabularCPD(variable=d, variable_card=2, values=[[0.9], [0.1]])
        )

    # Symptom CPDs — probability increases with number of active diseases
    for s in symptom_nodes:
        n_combos = 2 ** num_diseases
        false_vals = [
            max(0.05, 0.9 - 0.1 * bin(i).count("1")) for i in range(n_combos)
        ]
        true_vals = [1.0 - v for v in false_vals]
        model.add_cpds(
            TabularCPD(
                variable=s,
                variable_card=2,
                values=[false_vals, true_vals],
                evidence=disease_nodes,
                evidence_card=[2] * num_diseases,
            )
        )

    # Treatment — driven by first disease only
    model.add_cpds(
        TabularCPD(
            variable="Treatment",
            variable_card=2,
            values=[[0.95, 0.05], [0.05, 0.95]],
            evidence=[disease_nodes[0]],
            evidence_card=[2],
        )
    )

    # Recovery — depends on all diseases + Treatment
    recovery_parents = disease_nodes + ["Treatment"]
    n_combos = 2 ** len(recovery_parents)
    true_vals = [max(0.1, 0.99 - 0.1 * bin(i).count("1")) for i in range(n_combos)]
    false_vals = [1.0 - v for v in true_vals]
    model.add_cpds(
        TabularCPD(
            variable="Recovery",
            variable_card=2,
            values=[false_vals, true_vals],
            evidence=recovery_parents,
            evidence_card=[2] * len(recovery_parents),
        )
    )

    assert model.check_model(), "Expanded model is invalid."
    return model


def time_variable_elimination(model: BayesianNetwork) -> float:
    """Return the wall-clock seconds taken by Variable Elimination on *model*."""
    symptom_nodes = sorted(n for n in model.nodes() if n.startswith("Symptom"))
    if not symptom_nodes or "Treatment" not in model.nodes():
        raise ValueError("Model must contain at least one Symptom node and Treatment.")

    start = time.perf_counter()
    ve = VariableElimination(model)
    ve.query(
        variables=["Recovery"],
        evidence={symptom_nodes[0]: 1, "Treatment": 1},
    )
    return time.perf_counter() - start


def time_likelihood_weighted_sampling(
    model: BayesianNetwork,
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Return the wall-clock seconds taken by likelihood-weighted sampling."""
    symptom_nodes = sorted(n for n in model.nodes() if n.startswith("Symptom"))
    if not symptom_nodes or "Treatment" not in model.nodes():
        raise ValueError("Model must contain at least one Symptom node and Treatment.")

    start = time.perf_counter()
    sampler = BayesianModelSampling(model)
    sampler.likelihood_weighted_sample(
        evidence=[(symptom_nodes[0], 1), ("Treatment", 1)],
        size=sample_size,
        seed=seed,
    )
    return time.perf_counter() - start


def plot_runtime_comparison(
    network_sizes: Optional[List[int]] = None,
) -> Tuple[List[int], List[Optional[float]], List[Optional[float]]]:
    """
    Benchmark both inference methods across increasing network sizes and
    produce side-by-side runtime plots.

    Returns
    -------
    network_sizes, ve_times, lw_times
    """
    if network_sizes is None:
        network_sizes = [2, 4, 6, 8, 10, 12]

    ve_times: List[Optional[float]] = []
    lw_times: List[Optional[float]] = []

    for size in network_sizes:
        print(f"Benchmarking size {size}...")
        model = build_expanded_model(num_diseases=size, num_symptoms=size)

        try:
            ve_times.append(time_variable_elimination(model))
        except Exception as exc:
            print(f"  Variable Elimination failed: {exc}")
            ve_times.append(None)

        try:
            lw_times.append(time_likelihood_weighted_sampling(model))
        except Exception as exc:
            print(f"  Likelihood Weighted Sampling failed: {exc}")
            lw_times.append(None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(network_sizes, ve_times, marker="o")
    axes[0].set_title("Variable Elimination Runtime vs Network Size")
    axes[0].set_xlabel("Network Size (N diseases = N symptoms)")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].grid(True)

    axes[1].plot(network_sizes, lw_times, marker="o", color="orange")
    axes[1].set_title("Likelihood Weighted Sampling Runtime vs Network Size")
    axes[1].set_xlabel("Network Size (N diseases = N symptoms)")
    axes[1].set_ylabel("Execution Time (seconds)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    return network_sizes, ve_times, lw_times
