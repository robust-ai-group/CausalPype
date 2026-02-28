import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class ArrowStrength(BaseTask):
    """Compute the causal strength of each edge pointing into a target node."""
    name = "arrow_strength"

    def __init__(self, target, difference_estimation_func=None):
        self.target = target
        self.difference_estimation_func = difference_estimation_func

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.target)

        kw = {}
        if self.difference_estimation_func:
            kw["difference_estimation_func"] = self.difference_estimation_func

        strengths = gcm.arrow_strength(model.scm, target_node=self.target, **kw)

        strengths_clean = {f"{k[0]} -> {k[1]}": float(v) for k, v in strengths.items()}

        return TaskResult(
            task_name="Arrow Strength",
            estimate=strengths_clean,
            details={
                "target": self.target,
                "strengths": strengths_clean,
                "raw_strengths": strengths,
            },
        )