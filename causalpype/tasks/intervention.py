import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class Intervention(BaseTask):
    """do-calculus style intervention using gcm.interventional_samples."""
    name = "intervention"

    def __init__(self, interventions, outcome=None, num_samples=2000):
        """
        interventions: dict mapping node -> value or callable.
            e.g. {"X": 1.0} or {"X": lambda x: x + 1}
        outcome: optional node to focus results on. If None, returns all nodes.
        """
        self.interventions = interventions
        self.outcome = outcome
        self.num_samples = num_samples

    def _to_callable(self, v):
        if callable(v):
            return v
        return lambda x: v

    def run(self, model, **kwargs):
        self.validate(model)
        for node in self.interventions:
            self._check_node(model, node)
        if self.outcome:
            self._check_node(model, self.outcome)

        intervention_fns = {k: self._to_callable(v) for k, v in self.interventions.items()}

        samples = gcm.interventional_samples(
            model.scm,
            interventions=intervention_fns,
            num_samples_to_draw=self.num_samples,
        )

        details = {
            "interventions": {k: v if not callable(v) else repr(v) for k, v in self.interventions.items()},
            "samples": samples,
        }

        if self.outcome:
            estimate = float(samples[self.outcome].mean())
            details["outcome"] = self.outcome
            details["mean"] = estimate
            details["std"] = float(samples[self.outcome].std())
        else:
            estimate = {col: float(samples[col].mean()) for col in samples.columns}

        return TaskResult(
            task_name="Intervention",
            estimate=estimate,
            details=details,
        )