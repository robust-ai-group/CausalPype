import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class DistributionChange(BaseTask):
    """Attribute the change in a target's distribution between two datasets
    to individual nodes in the causal graph."""
    name = "distribution_change"

    def __init__(self, target, old_data, new_data, num_samples=2000):
        self.target = target
        self.old_data = old_data
        self.new_data = new_data
        self.num_samples = num_samples

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.target)

        contributions = gcm.distribution_change(
            model.scm,
            old_data=self.old_data,
            new_data=self.new_data,
            target_node=self.target,
            num_samples=self.num_samples,
        )

        contributions_clean = {str(k): float(v) for k, v in contributions.items()}

        return TaskResult(
            task_name="Distribution Change",
            estimate=contributions_clean,
            details={
                "target": self.target,
                "contributions": contributions_clean,
                "n_old": len(self.old_data),
                "n_new": len(self.new_data),
            },
        )