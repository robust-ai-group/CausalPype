from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TaskResult:
    task_name: str
    estimate: Any = None
    details: dict = field(default_factory=dict)

    def to_dict(self):
        clean = {}
        for k, v in self.details.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = list(v)
            elif isinstance(v, dict):
                clean[k] = {str(dk): (float(dv) if isinstance(dv, (np.floating, float)) else dv)
                            for dk, dv in v.items()
                            if isinstance(dv, (int, float, str, bool, type(None), np.floating))}
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif isinstance(v, pd.DataFrame):
                clean[k] = v.to_dict(orient="records")
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            # Skip non-serializable objects (estimators, raw gcm objects, etc.)

        estimate = self.estimate
        if isinstance(estimate, np.floating):
            estimate = float(estimate)
        elif isinstance(estimate, pd.DataFrame):
            estimate = estimate.to_dict(orient="records")
        elif isinstance(estimate, dict):
            estimate = {str(k): (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in estimate.items()}

        return {
            "task_name": self.task_name,
            "estimate": estimate,
            "details": clean,
        }

    def __repr__(self):
        est = f"{self.estimate:.6f}" if isinstance(self.estimate, float) else str(self.estimate)
        return f"TaskResult({self.task_name}: {est})"

    def __str__(self):
        from ..display import format_result
        return format_result(self)

    def summary(self) -> str:
        """Return a rich, task-specific formatted string."""
        return str(self)


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