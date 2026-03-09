import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class _IndexedIntervention:
    """Stateful callable that returns per-sample values from a pre-computed array.

    DoWhy's counterfactual_samples calls intervention functions once per sample
    with a scalar input. This class tracks the sample index to return the
    correct pre-computed value for each individual.
    """
    def __init__(self, values):
        self.values = np.asarray(values)
        self.idx = 0

    def __call__(self, x):
        val = float(self.values[self.idx])
        self.idx += 1
        return val


class Mediation(BaseTask):
    """Natural direct/indirect effect decomposition using Pearl's mediation formula.

    NDE = E[Y(treatment, M(control))] - E[Y(control, M(control))]
    NIE = E[Y(treatment, M(treatment))] - E[Y(treatment, M(control))]
    TE  = NDE + NIE
    """
    name = "mediation"

    def __init__(self, treatment, outcome, mediators=None,
                 treatment_value=1, control_value=0):
        self.treatment = treatment
        self.outcome = outcome
        self.mediators = mediators
        self.treatment_value = treatment_value
        self.control_value = control_value

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
        observed = model.data

        # Y(control, M(control))
        cf_control = gcm.counterfactual_samples(
            model.scm,
            interventions={self.treatment: lambda x: self.control_value},
            observed_data=observed,
        )

        # Y(treatment, M(treatment))
        cf_treatment = gcm.counterfactual_samples(
            model.scm,
            interventions={self.treatment: lambda x: self.treatment_value},
            observed_data=observed,
        )

        # Y(treatment, M(control)) — nested counterfactual
        # Force each mediator to its individual-level natural value under control.
        if mediators:
            nde_interventions = {self.treatment: lambda x: self.treatment_value}
            for med in mediators:
                nde_interventions[med] = _IndexedIntervention(cf_control[med].values)

            cf_nde = gcm.counterfactual_samples(
                model.scm,
                interventions=nde_interventions,
                observed_data=observed,
            )

            # Verify that all indexed interventions were consumed in order.
            # If DoWhy changes its internal iteration order, this will catch it.
            for med in mediators:
                interv = nde_interventions[med]
                if isinstance(interv, _IndexedIntervention):
                    assert interv.idx == len(interv.values), (
                        f"_IndexedIntervention for '{med}' consumed {interv.idx} of "
                        f"{len(interv.values)} values. DoWhy's sample iteration order "
                        f"may have changed — mediation results are unreliable."
                    )
        else:
            cf_nde = cf_treatment

        y_control = cf_control[self.outcome].values
        y_treatment = cf_treatment[self.outcome].values
        y_nde = cf_nde[self.outcome].values

        total_effect = float(np.mean(y_treatment) - np.mean(y_control))
        nde = float(np.mean(y_nde) - np.mean(y_control))
        nie = float(np.mean(y_treatment) - np.mean(y_nde))
        proportion_mediated = nie / total_effect if total_effect != 0 else None

        individual_nde = y_nde - y_control
        individual_nie = y_treatment - y_nde

        return TaskResult(
            task_name="Mediation",
            estimate=nie,
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "mediators": mediators,
                "total_effect": total_effect,
                "natural_direct_effect": nde,
                "natural_indirect_effect": nie,
                "proportion_mediated": proportion_mediated,
                "individual_nde": individual_nde,
                "individual_nie": individual_nie,
            },
        )
