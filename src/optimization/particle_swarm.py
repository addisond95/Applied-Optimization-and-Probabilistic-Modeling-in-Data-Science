from typing import Dict, Optional

import numpy as np
import pyswarms as ps


def _objective(x: np.ndarray) -> np.ndarray:
    """Objective: minimise -4*x1 - 3*x2 (i.e. maximise 4*x1 + 3*x2)."""
    return -4 * x[:, 0] - 3 * x[:, 1]


def _is_feasible(x: np.ndarray) -> np.ndarray:
    c1 = x[:, 0] + 2 * x[:, 1] <= 8
    c2 = 3 * x[:, 0] + x[:, 1] <= 9
    c3 = x[:, 0] >= 0
    c4 = x[:, 1] >= 0
    return c1 & c2 & c3 & c4


def _penalized_objective(x: np.ndarray) -> np.ndarray:
    penalty = 1e6
    obj = _objective(x)
    obj[~_is_feasible(x)] += penalty
    return obj


def solve_pso(
    n_particles: int = 50,
    iters: int = 100,
    options: Optional[Dict] = None,
) -> Dict:
    """
    Approximate the same LP as solve_lp() using Particle Swarm Optimisation.

    Returns a dict with keys: x1, x2, objective, success.
    """
    if options is None:
        options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(_penalized_objective, iters=iters)

    return {
        "x1": best_pos[0],
        "x2": best_pos[1],
        "objective": best_cost,
        "success": True,
    }
