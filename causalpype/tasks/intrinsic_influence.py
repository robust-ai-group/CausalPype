import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class IntrinsicCausalInfluence(BaseTask):
    """Attribute variance in a target to upstream noise terms via Shapley values."""
    name = "intrinsic_causal_influence"

    def __init__(self, target, prediction_model="approx"):
        self.target = target
        self.prediction_model = prediction_model

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.target)

        influences = gcm.intrinsic_causal_influence(
            model.scm,
            target_node=self.target,
            prediction_model=self.prediction_model,
        )

        influences_clean = {str(k): float(v) for k, v in influences.items()}

        total = sum(influences_clean.values())
        normalized = {k: v / total if total > 0 else 0 for k, v in influences_clean.items()}

        return TaskResult(
            task_name="Intrinsic Causal Influence",
            estimate=influences_clean,
            details={
                "target": self.target,
                "influences": influences_clean,
                "normalized": normalized,
                "total_variance_explained": total,
            },
        )