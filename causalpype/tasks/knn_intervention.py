import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .base import BaseTask, TaskResult


class KNNIntervention(BaseTask):
    """Estimate individual treatment effects by matching each unit
    to its K nearest neighbors with the opposite treatment value."""
    name = "knn_intervention"

    def __init__(self, treatment, outcome, k=5, match_on=None,
                 treatment_value=1, control_value=0):
        self.treatment = treatment
        self.outcome = outcome
        self.k = k
        self.match_on = match_on
        self.treatment_value = treatment_value
        self.control_value = control_value

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        data = model.data

        if self.match_on:
            match_cols = self.match_on
        else:
            match_cols = [c for c in data.columns if c not in (self.treatment, self.outcome)]

        treated_mask = data[self.treatment] == self.treatment_value
        control_mask = data[self.treatment] == self.control_value

        treated = data[treated_mask]
        control = data[control_mask]

        if len(treated) == 0:
            raise ValueError(
                f"No units found with {self.treatment}=={self.treatment_value}. "
                f"Unique values: {sorted(data[self.treatment].unique()[:10])}"
            )
        if len(control) == 0:
            raise ValueError(
                f"No units found with {self.treatment}=={self.control_value}. "
                f"Unique values: {sorted(data[self.treatment].unique()[:10])}"
            )

        X_treated_raw = treated[match_cols].values
        X_control_raw = control[match_cols].values
        Y_treated = treated[self.outcome].values
        Y_control = control[self.outcome].values

        # Standardize covariates so distance is meaningful across different scales
        scaler = StandardScaler()
        scaler.fit(data[match_cols].values)
        X_treated = scaler.transform(X_treated_raw)
        X_control = scaler.transform(X_control_raw)

        nn_control = NearestNeighbors(n_neighbors=min(self.k, len(control))).fit(X_control)
        dist_t, idx_t = nn_control.kneighbors(X_treated)
        cf_treated = np.mean(Y_control[idx_t], axis=1)
        ite_treated = Y_treated - cf_treated

        nn_treated = NearestNeighbors(n_neighbors=min(self.k, len(treated))).fit(X_treated)
        dist_c, idx_c = nn_treated.kneighbors(X_control)
        cf_control = np.mean(Y_treated[idx_c], axis=1)
        ite_control = cf_control - Y_control

        all_ite = np.concatenate([ite_treated, ite_control])

        return TaskResult(
            task_name="KNN Intervention",
            estimate=float(np.mean(all_ite)),
            details={
                "treatment": self.treatment,
                "outcome": self.outcome,
                "k": self.k,
                "match_on": match_cols,
                "ate": float(np.mean(all_ite)),
                "att": float(np.mean(ite_treated)),
                "atc": float(np.mean(ite_control)),
                "std_ite": float(np.std(all_ite)),
                "match_quality_treated": float(np.mean(dist_t)),
                "match_quality_control": float(np.mean(dist_c)),
                "n_treated": len(treated),
                "n_control": len(control),
                "ite_treated": ite_treated,
                "ite_control": ite_control,
                "all_ite": all_ite,
            },
        )