"""
scripts/run_optimization.py
============================
Solve the constrained LP problem with both scipy linprog (exact) and
pyswarms PSO (approximate), then compare results.

Objective:  min  -4*x1 - 3*x2
Subject to: x1 + 2*x2 <= 8
            3*x1 +  x2 <= 9
            x1, x2 >= 0

Run from the repository root:
    python scripts/run_optimization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.linear_programming import solve_lp
from src.optimization.particle_swarm import solve_pso


def main() -> None:
    # ------------------------------------------------------------------
    # Linear Programming (exact)
    # ------------------------------------------------------------------
    print("Solving with Linear Programming (scipy HiGHS) …")
    lp = solve_lp()
    if lp["success"]:
        print(f"  x1 = {lp['x1']:.6f}")
        print(f"  x2 = {lp['x2']:.6f}")
        print(f"  f(x*) = {lp['objective']:.6f}")
    else:
        print(f"  LP failed: {lp['message']}")

    # ------------------------------------------------------------------
    # Particle Swarm Optimisation (approximate)
    # ------------------------------------------------------------------
    print("\nSolving with Particle Swarm Optimisation (pyswarms) …")
    pso = solve_pso(n_particles=50, iters=100)
    print(f"  x1 = {pso['x1']:.6f}")
    print(f"  x2 = {pso['x2']:.6f}")
    print(f"  f(x*) ≈ {pso['objective']:.6f}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    if lp["success"]:
        print("\n--- Comparison ---")
        print(f"  |Δx1|       = {abs(lp['x1'] - pso['x1']):.2e}")
        print(f"  |Δx2|       = {abs(lp['x2'] - pso['x2']):.2e}")
        print(f"  |Δf(x*)|    = {abs(lp['objective'] - pso['objective']):.2e}")


if __name__ == "__main__":
    main()
