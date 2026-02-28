import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class FairnessAudit(BaseTask):
    """Counterfactual fairness: would outcomes change if the protected
    attribute had been different, for observed individuals?"""
    name = "fairness_audit"

    def __init__(self, protected_attribute, outcome, privileged_value=1, unprivileged_value=0):
        self.protected_attribute = protected_attribute
        self.outcome = outcome
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.protected_attribute)
        self._check_node(model, self.outcome)

        data = model.data

        # Counterfactual: what if everyone had privileged value?
        cf_priv = gcm.counterfactual_samples(
            model.scm,
            interventions={self.protected_attribute: lambda x: self.privileged_value},
            observed_data=data,
        )

        # Counterfactual: what if everyone had unprivileged value?
        cf_unpriv = gcm.counterfactual_samples(
            model.scm,
            interventions={self.protected_attribute: lambda x: self.unprivileged_value},
            observed_data=data,
        )

        y_cf_priv = cf_priv[self.outcome].values
        y_cf_unpriv = cf_unpriv[self.outcome].values
        disparity = float(np.mean(y_cf_priv) - np.mean(y_cf_unpriv))

        # Per-individual counterfactual unfairness
        individual_unfairness = y_cf_priv - y_cf_unpriv

        # Observational gap for comparison
        priv_mask = data[self.protected_attribute] == self.privileged_value
        unpriv_mask = data[self.protected_attribute] == self.unprivileged_value
        obs_gap = None
        if priv_mask.any() and unpriv_mask.any():
            obs_gap = float(data.loc[priv_mask, self.outcome].mean() -
                           data.loc[unpriv_mask, self.outcome].mean())

        return TaskResult(
            task_name="Fairness Audit",
            estimate=disparity,
            details={
                "protected_attribute": self.protected_attribute,
                "outcome": self.outcome,
                "counterfactual_disparity": disparity,
                "observational_gap": obs_gap,
                "mean_individual_unfairness": float(np.mean(np.abs(individual_unfairness))),
                "max_individual_unfairness": float(np.max(np.abs(individual_unfairness))),
                "n_privileged": int(priv_mask.sum()),
                "n_unprivileged": int(unpriv_mask.sum()),
            },
        )