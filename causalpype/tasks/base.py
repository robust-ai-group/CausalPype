from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Formatting helpers ──────────────────────────────────

_WIDTH = 58
_ACRONYMS = {"ate", "att", "atc", "nde", "nie", "ite", "cate", "std", "sd", "knn"}


def _title(name: str) -> str:
    """Centered title + double rule."""
    return f"{name.center(_WIDTH)}\n{'=' * _WIDTH}"


def _sep() -> str:
    return "-" * _WIDTH


def _end() -> str:
    return "=" * _WIDTH


def _fmt(value) -> str:
    """Format a scalar value for display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if value is None:
        return "-"
    return str(value)


def _kv(label: str, value) -> str:
    """Key-value row, value right-aligned to _WIDTH."""
    val_str = _fmt(value)
    padding = _WIDTH - 1 - len(label) - len(val_str)
    return f" {label}{' ' * max(padding, 1)}{val_str}"


def _label(key: str) -> str:
    """Convert snake_case to Title Case, preserving acronyms."""
    words = key.split("_")
    return " ".join(w.upper() if w.lower() in _ACRONYMS else w.title() for w in words)


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

    def _format(self) -> str:
        """Default statsmodels-style formatting. Override in subclasses."""
        lines = [_title(f"{self.task_name} Results")]
        for k, v in self.details.items():
            if isinstance(v, (bool, int, float, str, type(None), np.integer, np.floating)):
                lines.append(_kv(_label(k), v))
        lines.append(_sep())
        if isinstance(self.estimate, (int, float, np.integer, np.floating)):
            lines.append(_kv("Estimate", self.estimate))
        elif isinstance(self.estimate, str):
            lines.append(_kv("Result", self.estimate.upper()))
        elif isinstance(self.estimate, dict):
            for k, v in self.estimate.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    lines.append(_kv(f" {k}", v))
        else:
            lines.append(_kv("Estimate", str(self.estimate)))
        lines.append(_end())
        return "\n".join(lines)

    def __repr__(self):
        return self._format()

    def __str__(self):
        return self._format()

    def summary(self):
        """Print task-specific formatted summary."""
        print(self._format())


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