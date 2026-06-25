from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def build_medical_model() -> BayesianNetwork:
    """
    Construct the six-node Bayesian Network for medical diagnosis:

        Flu, COVID-19 → Fever
        Flu, COVID-19 → Cough
        COVID-19      → Treatment
        Flu, COVID-19, Treatment → Recovery

    Conditional probability tables match the problem specification.
    """
    model = BayesianNetwork(
        [
            ("Flu", "Fever"),
            ("Flu", "Cough"),
            ("COVID-19", "Fever"),
            ("COVID-19", "Cough"),
            ("COVID-19", "Treatment"),
            ("Flu", "Recovery"),
            ("COVID-19", "Recovery"),
            ("Treatment", "Recovery"),
        ]
    )

    cpd_flu = TabularCPD(
        variable="Flu", variable_card=2, values=[[0.88], [0.12]]
    )
    cpd_covid = TabularCPD(
        variable="COVID-19", variable_card=2, values=[[0.92], [0.08]]
    )

    # Columns: Flu=0/COVID=0, Flu=0/COVID=1, Flu=1/COVID=0, Flu=1/COVID=1
    cpd_fever = TabularCPD(
        variable="Fever",
        variable_card=2,
        values=[
            [0.99, 0.15, 0.10, 0.02],  # P(Fever=False)
            [0.01, 0.85, 0.90, 0.98],  # P(Fever=True)
        ],
        evidence=["Flu", "COVID-19"],
        evidence_card=[2, 2],
    )
    cpd_cough = TabularCPD(
        variable="Cough",
        variable_card=2,
        values=[
            [0.98, 0.40, 0.30, 0.15],
            [0.02, 0.60, 0.70, 0.85],
        ],
        evidence=["Flu", "COVID-19"],
        evidence_card=[2, 2],
    )
    cpd_treatment = TabularCPD(
        variable="Treatment",
        variable_card=2,
        values=[[0.95, 0.05], [0.05, 0.95]],
        evidence=["COVID-19"],
        evidence_card=[2],
    )

    # Columns ordered by (Flu, COVID-19, Treatment) binary combinations
    cpd_recovery = TabularCPD(
        variable="Recovery",
        variable_card=2,
        values=[
            # P(Recovery=False) — 8 parent combinations
            [
                1 - 0.99, 1 - 0.99,  # Flu=0, COVID=0, Trt=0/1
                1 - 0.90, 1 - 0.50,  # Flu=0, COVID=1, Trt=1/0
                1 - 0.85, 1 - 0.85,  # Flu=1, COVID=0, Trt=0/1
                1 - 0.80, 1 - 0.30,  # Flu=1, COVID=1, Trt=1/0
            ],
            # P(Recovery=True)
            [0.99, 0.99, 0.90, 0.50, 0.85, 0.85, 0.80, 0.30],
        ],
        evidence=["Flu", "COVID-19", "Treatment"],
        evidence_card=[2, 2, 2],
    )

    model.add_cpds(
        cpd_flu, cpd_covid, cpd_fever, cpd_cough, cpd_treatment, cpd_recovery
    )
    assert model.check_model(), "Bayesian Network is invalid."
    return model
