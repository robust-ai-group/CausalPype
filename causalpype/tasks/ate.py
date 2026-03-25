import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult, _title, _sep, _end, _kv


class ATEResult(TaskResult):
    def _format(self) -> str:
        d = self.details
        lines = [
            _title("ATE Results"),
            _kv("Treatment", d["treatment"]),
            _kv("Outcome", d["outcome"]),
            _kv("Treatment Value", d["treatment_value"]),
            _kv("Control Value", d["control_value"]),
            _sep(),
            _kv("Estimate", self.estimate),
            _kv("Num Samples", d["num_samples"]),
            _end(),
        ]
        return "\n".join(lines)


class ATE(BaseTask):
    name = "ate"

    def __init__(self, treatment, outcome, treatment_value=1, control_value=0, num_samples=2000):
        self.treatment = treatment
        self.outcome = outcome
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.num_samples = num_samples

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        effect = gcm.average_causal_effect(
            model.scm,
            target_node=self.outcome,
            interventions_alternative={self.treatment: lambda x: self.treatment_value},
            interventions_reference={self.treatment: lambda x: self.control_value},
            num_samples_to_draw=self.num_samples,
        )

        return ATEResult(
            task_name="ATE",
            estimate=float(effect),
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "treatment_value": self.treatment_value,
                "control_value": self.control_value,
                "num_samples": self.num_samples,
            },
        )