from typing import Dict

import numpy as np
from scipy.optimize import linprog


def solve_lp() -> Dict:
    """
    Solve the linear programme:

        min  -4*x1 - 3*x2
        s.t. x1 + 2*x2 <= 8
             3*x1 +  x2 <= 9
             x1, x2 >= 0

    Returns a dict with keys: x1, x2, objective, success.
    """
    c = [-4, -3]
    A_ub = [[1, 2], [3, 1]]
    b_ub = [8, 9]
    bounds = [(0, None), (0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        return {
            "x1": result.x[0],
            "x2": result.x[1],
            "objective": result.fun,
            "success": True,
        }
    return {"success": False, "message": result.message}
