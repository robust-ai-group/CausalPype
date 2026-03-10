import numpy as np
import pandas as pd
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class CausalEffectCurve(BaseTask):
    """Estimate E[Y|do(X=x)] for a range of treatment values using the SCM."""
    name = "causal_effect_curve"

    def __init__(self, treatment, outcome, treatment_values=None,
                 n_points=20, num_samples=1000):
        self.treatment = treatment
        self.outcome = outcome
        self.treatment_values = treatment_values
        self.n_points = n_points
        self.num_samples = num_samples

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        treatment_values = self.treatment_values
        if treatment_values is None:
            t_min = model.data[self.treatment].min()
            t_max = model.data[self.treatment].max()
            treatment_values = np.linspace(t_min, t_max, self.n_points)

        responses = []
        for t_val in treatment_values:
            samples = gcm.interventional_samples(
                model.scm,
                interventions={self.treatment: lambda x, v=t_val: v},
                num_samples_to_draw=self.num_samples,
            )
            y_vals = samples[self.outcome].values
            responses.append({
                "treatment_value": float(t_val),
                "expected_outcome": float(np.mean(y_vals)),
                "std": float(np.std(y_vals)),
            })

        response_df = pd.DataFrame(responses)

        return TaskResult(
            task_name="Causal Effect Curve",
            estimate=response_df,
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "responses": responses,
                "response_df": response_df,
            },
        )