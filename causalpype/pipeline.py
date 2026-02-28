from typing import List, Union
from .tasks.base import BaseTask, TaskResult


class Pipeline:
    def __init__(self, model):
        self.model = model
        self.results = []

    def run(self, tasks: Union[BaseTask, List[BaseTask]], **kwargs) -> Union[TaskResult, List[TaskResult]]:
        if isinstance(tasks, BaseTask):
            tasks = [tasks]

        self.results = []
        for task in tasks:
            result = task.run(self.model, **kwargs)
            self.results.append(result)

        return self.results[0] if len(self.results) == 1 else self.results

    def summary(self):
        lines = ["=" * 60, "CausalPype Pipeline Results", "=" * 60]
        for r in self.results:
            lines.append(f"\n--- {r.task_name} ---")
            if isinstance(r.estimate, float):
                lines.append(f"  Estimate: {r.estimate:.6f}")
            else:
                lines.append(f"  Estimate: {r.estimate}")

            skip = {"samples", "individual_effects", "predictions", "residuals",
                    "anomalies", "estimator", "cate_model", "policy",
                    "individual_counterfactuals", "all_ite", "ite_treated",
                    "ite_control", "response_df", "assignments",
                    "counterfactual_samples", "interventional_samples",
                    "noise_data", "observed_data"}
            for k, v in r.details.items():
                if k in skip:
                    continue
                lines.append(f"  {k}: {v}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return f"Pipeline(model={self.model}, tasks_run={len(self.results)})"