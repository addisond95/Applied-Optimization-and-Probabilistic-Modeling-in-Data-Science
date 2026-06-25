from typing import Dict, List

import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def exact_inference(
    model: BayesianNetwork,
    variables: List[str],
    evidence: Dict[str, int],
):
    """
    Perform exact probabilistic inference via Variable Elimination.

    Parameters
    ----------
    model     : fitted BayesianNetwork
    variables : query variables
    evidence  : observed variables and their values (0 = False, 1 = True)

    Returns
    -------
    pgmpy DiscreteFactor with the posterior distribution.
    """
    ve = VariableElimination(model)
    return ve.query(variables=variables, evidence=evidence)


def approximate_inference(
    model: BayesianNetwork,
    variables: List[str],
    evidence: Dict[str, int],
    size: int = 5000,
    seed: int = 42,
) -> Dict[str, pd.Series]:
    """
    Approximate inference via likelihood-weighted sampling.

    Returns
    -------
    Dict mapping each query variable to its empirical probability distribution.
    """
    sampler = BayesianModelSampling(model)
    samples = sampler.likelihood_weighted_sample(
        evidence=list(evidence.items()),
        size=size,
        seed=seed,
    )
    return {
        var: samples[var].value_counts(normalize=True).sort_index()
        for var in variables
    }
