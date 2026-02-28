import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class Mediation(BaseTask):
    """Natural direct/indirect effect decomposition using counterfactuals from the SCM."""
    name = "mediation"

    def __init__(self, treatment, outcome, mediators=None,
                 treatment_value=1, control_value=0, num_samples=2000):
        self.treatment = treatment
        self.outcome = outcome
        self.mediators = mediators
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.num_samples = num_samples

    def _find_mediators(self, model):
        paths = model.get_all_paths(self.treatment, self.outcome)
        mediators = set()
        for path in paths:
            for node in path[1:-1]:
                mediators.add(node)
        return list(mediators)

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        mediators = self.mediators or self._find_mediators(model)

        # Total effect: do(T=1) - do(T=0)
        y_treat = gcm.interventional_samples(
            model.scm,
            {self.treatment: lambda x: self.treatment_value},
            num_samples_to_draw=self.num_samples,
        )[self.outcome].values

        y_control = gcm.interventional_samples(
            model.scm,
            {self.treatment: lambda x: self.control_value},
            num_samples_to_draw=self.num_samples,
        )[self.outcome].values

        total_effect = float(np.mean(y_treat) - np.mean(y_control))

        # Controlled direct effect: do(T=1, M=m*) - do(T=0, M=m*)
        # where m* is the natural value of M under control
        if mediators:
            m_control = gcm.interventional_samples(
                model.scm,
                {self.treatment: lambda x: self.control_value},
                num_samples_to_draw=self.num_samples,
            )

            # Direct effect: intervene on T while holding mediators at control values
            med_interventions = {self.treatment: lambda x: self.treatment_value}
            for med in mediators:
                med_mean = float(m_control[med].mean())
                med_interventions[med] = lambda x, v=med_mean: v

            y_direct = gcm.interventional_samples(
                model.scm,
                med_interventions,
                num_samples_to_draw=self.num_samples,
            )[self.outcome].values

            direct_effect = float(np.mean(y_direct) - np.mean(y_control))
            indirect_effect = total_effect - direct_effect
        else:
            direct_effect = total_effect
            indirect_effect = 0.0

        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else None

        return TaskResult(
            task_name="Mediation",
            estimate=indirect_effect,
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "mediators": mediators,
                "total_effect": total_effect,
                "direct_effect": direct_effect,
                "indirect_effect": indirect_effect,
                "proportion_mediated": proportion_mediated,
            },
        )