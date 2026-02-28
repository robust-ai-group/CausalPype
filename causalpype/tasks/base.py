from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    task_name: str
    estimate: Any = None
    details: dict = field(default_factory=dict)

    def __repr__(self):
        est = f"{self.estimate:.6f}" if isinstance(self.estimate, float) else str(self.estimate)
        return f"TaskResult({self.task_name}: {est})"


class BaseTask(ABC):
    name: str = "base"

    def validate(self, model):
        model._check_fitted()

    def _check_node(self, model, node):
        if node not in model.graph.nodes:
            raise ValueError(f"Node '{node}' not in graph. Available: {list(model.graph.nodes)}")

    @abstractmethod
    def run(self, model, **kwargs) -> TaskResult:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"