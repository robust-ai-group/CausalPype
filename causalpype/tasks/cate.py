import logging
import numpy as np
from .base import BaseTask, TaskResult

logger = logging.getLogger(__name__)


class CATE(BaseTask):
    """Heterogeneous treatment effects using EconML estimators."""
    name = "cate"

    def __init__(self, treatment, outcome, effect_modifiers, confounders=None,
                 method="linear_dml", estimator=None, **estimator_kwargs):
        self.treatment = treatment
        self.outcome = outcome
        self.effect_modifiers = effect_modifiers
        self.confounders = confounders
        self.method = method
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs

    def _build_estimator(self):
        if self.estimator is not None:
            return self.estimator
        if self.method == "linear_dml":
            from econml.dml import LinearDML
            return LinearDML(**self.estimator_kwargs)
        elif self.method == "causal_forest":
            from econml.dml import CausalForestDML
            return CausalForestDML(**self.estimator_kwargs)
        elif self.method == "metalearner":
            from econml.metalearners import TLearner
            from sklearn.ensemble import GradientBoostingRegressor
            return TLearner(models=GradientBoostingRegressor(), **self.estimator_kwargs)
        raise ValueError(f"Unknown CATE method: {self.method}")

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        data = model.data
        T = data[self.treatment].values.ravel()
        Y = data[self.outcome].values.ravel()
        X = data[self.effect_modifiers].values

        if self.confounders:
            W = data[self.confounders].values
        else:
            auto_confounders = model.get_adjustment_set(self.treatment, self.outcome)
            W = data[auto_confounders].values if auto_confounders else None

        est = self._build_estimator()

        if self.method == "metalearner":
            # Metalearners don't have a separate W argument, so we include
            # confounders in the feature matrix for fitting. However, we
            # compute effects on X (effect modifiers) only, so the returned
            # heterogeneity is w.r.t. the effect modifiers, not confounders.
            if W is not None:
                X_fit = np.hstack([X, W])
            else:
                X_fit = X
            est.fit(Y, T, X=X_fit)
            effects = est.effect(X_fit)
        else:
            est.fit(Y, T, X=X, W=W)
            effects = est.effect(X)

        details = {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "effect_modifiers": self.effect_modifiers,
            "method": self.method,
            "mean_effect": float(np.mean(effects)),
            "std_effect": float(np.std(effects)),
            "individual_effects": effects,
            "bounds": (float(np.min(effects)), float(np.max(effects))),
            "estimator": est,
        }

        try:
            lb, ub = est.effect_interval(X)
            details["lower_bound"] = lb
            details["upper_bound"] = ub
        except Exception:
            logger.debug("Confidence intervals not available for method '%s'", self.method)

        return TaskResult(
            task_name="CATE",
            estimate=float(np.mean(effects)),
            details=details,
        )