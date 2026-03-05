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

        return self.results

    def summary(self):
        lines = ["=" * 60, "CausalPype Pipeline Results", "=" * 60]
        for r in self.results:
            lines.append("")
            lines.append(str(r))
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return f"Pipeline(model={self.model}, tasks_run={len(self.results)})"