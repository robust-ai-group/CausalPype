import numpy as np
import pandas as pd
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult, _title, _sep, _end, _kv


class CounterfactualResult(TaskResult):
    def _format(self) -> str:
        d = self.details
        interv_str = ", ".join(f"{k} := {v}" for k, v in d["interventions"].items())
        lines = [
            _title("Counterfactual Results"),
            _kv("Interventions", interv_str),
            _kv("N Units", d["n_units"]),
        ]
        if "outcome" in d:
            lines.append(_kv("Outcome", d["outcome"]))
            lines.append(_sep())
            lines.append(_kv("Factual Mean", d["factual_mean"]))
            lines.append(_kv("Counterfactual Mean", d["counterfactual_mean"]))
            lines.append(_kv("Mean Effect", d["mean_effect"]))
        else:
            lines.append(_sep())
            for k, v in self.estimate.items():
                lines.append(_kv(f" {k}", v))
        lines.append(_end())
        return "\n".join(lines)


class Counterfactual(BaseTask):
    """What would have happened if we intervened differently for observed units?"""
    name = "counterfactual"

    def __init__(self, interventions, observed_data=None, outcome=None):
        """
        interventions: dict mapping node -> value or callable.
        observed_data: DataFrame of factual observations. If None, uses model.data.
        outcome: optional node to focus on.
        """
        self.interventions = interventions
        self.observed_data = observed_data
        self.outcome = outcome

    def _to_callable(self, v):
        if callable(v):
            return v
        return lambda x: v

    def run(self, model, **kwargs):
        self.validate(model)
        for node in self.interventions:
            self._check_node(model, node)

        observed = self.observed_data if self.observed_data is not None else model.data
        intervention_fns = {k: self._to_callable(v) for k, v in self.interventions.items()}

        cf_samples = gcm.counterfactual_samples(
            model.scm,
            interventions=intervention_fns,
            observed_data=observed,
        )

        details = {
            "interventions": {k: v if not callable(v) else repr(v) for k, v in self.interventions.items()},
            "counterfactual_samples": cf_samples,
            "n_units": len(observed),
        }

        if self.outcome:
            self._check_node(model, self.outcome)
            factual = observed[self.outcome].values
            counterfactual = cf_samples[self.outcome].values
            details["outcome"] = self.outcome
            details["factual_mean"] = float(np.mean(factual))
            details["counterfactual_mean"] = float(np.mean(counterfactual))
            details["individual_effects"] = counterfactual - factual
            details["mean_effect"] = float(np.mean(counterfactual - factual))
            estimate = details["counterfactual_mean"]
        else:
            estimate = {col: float(cf_samples[col].mean()) for col in cf_samples.columns}

        return CounterfactualResult(
            task_name="Counterfactual",
            estimate=estimate,
            details=details,
        )