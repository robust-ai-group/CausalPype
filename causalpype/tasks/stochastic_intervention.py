import numpy as np
import pandas as pd
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult, _title, _sep, _end, _kv


class StochasticInterventionResult(TaskResult):
    def _format(self) -> str:
        d = self.details
        lines = [
            _title("Stochastic Intervention Results"),
            _kv("Treatment", d["treatment"]),
            _kv("Outcome", d["outcome"]),
            _kv("Shift", d["shift"]),
            _sep(),
            _kv("E[Y|baseline]", d["E[Y|baseline]"]),
            _kv("E[Y|shifted]", d["E[Y|shifted]"]),
            _kv("Effect", d["effect"]),
            _end(),
        ]
        return "\n".join(lines)


class StochasticIntervention(BaseTask):
    """Shift the treatment distribution rather than hard intervention.
    For continuous: add shift to natural value.
    For binary: increase probability by shift amount."""
    name = "stochastic_intervention"

    def __init__(self, treatment, outcome, shift=0.2, num_samples=2000, seed=42):
        self.treatment = treatment
        self.outcome = outcome
        self.shift = shift
        self.num_samples = num_samples
        self.seed = seed

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        rng = np.random.RandomState(self.seed)
        T = model.data[self.treatment].values
        T_clean = pd.Series(T).dropna().unique()
        is_binary = len(T_clean) <= 2 and set(T_clean).issubset({0, 1})

        if is_binary:
            # Shift: with probability `shift`, flip 0->1
            def shifted_treatment(x):
                original_shape = x.shape
                result = x.copy().ravel()
                zeros = result == 0
                flips = rng.random(zeros.sum()) < self.shift
                result[np.where(zeros)[0][flips]] = 1
                return result.reshape(original_shape)
        else:
            def shifted_treatment(x):
                return x + self.shift

        # Baseline: no intervention
        baseline = gcm.interventional_samples(
            model.scm,
            interventions={},
            num_samples_to_draw=self.num_samples,
        )[self.outcome].values

        # Shifted intervention
        shifted = gcm.interventional_samples(
            model.scm,
            interventions={self.treatment: shifted_treatment},
            num_samples_to_draw=self.num_samples,
        )[self.outcome].values

        mean_baseline = float(np.mean(baseline))
        mean_shifted = float(np.mean(shifted))

        return StochasticInterventionResult(
            task_name="Stochastic Intervention",
            estimate=mean_shifted - mean_baseline,
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "shift": self.shift,
                "is_binary": is_binary,
                "E[Y|baseline]": mean_baseline,
                "E[Y|shifted]": mean_shifted,
                "effect": mean_shifted - mean_baseline,
            },
        )